"""Run the sync scripts against fake OS commands and temporary file trees."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / 'examples' / 'macos-icloud-sync'

FAKE_OS = r'''
import json, os, pathlib, sys, time
base = pathlib.Path(os.environ['FAKE_SYNC_ROOT'])
state_file = base / 'scenario.json'
s = json.loads(state_file.read_text())
cmd = pathlib.Path(sys.argv[0]).name
source = base / 'source'
if cmd == 'stat' and sys.argv[2] in ['%m', '%z']:
    info = pathlib.Path(sys.argv[-1]).stat()
    print(int(info.st_mtime) if sys.argv[2] == '%m' else info.st_size)
    sys.exit(0)
def save(): state_file.write_text(json.dumps(s))
def event(kind, name):
    with (base / 'events.jsonl').open('a') as out:
        out.write(json.dumps([kind, name, time.monotonic()]) + '\n')
if cmd == 'find':
    if s.get('scan_error'):
        print('scan refused', file=sys.stderr); sys.exit(1)
    for name in s['files']:
        sys.stdout.buffer.write(os.fsencode(source / name) + b'\0')
    sys.exit(0)
f = pathlib.Path(sys.argv[-1]); name = str(f.relative_to(source)); item = s['files'][name]
if cmd == 'stat':
    if item.get('stat_error'): sys.exit(1)
    ready = item.get('ready_at', float('inf')) <= time.monotonic()
    print('-' if ready else 'dataless'); sys.exit(0)
event(cmd, name)
if cmd == 'icloud-download' and s.get('helper_denied'):
    print('Error Domain=NSCocoaErrorDomain Code=257', file=sys.stderr); sys.exit(1)
if item.get('request_hang'): time.sleep(20)
if item.get('vanish'):
    f.unlink(missing_ok=True)
elif not item.get('stuck'):
    item.setdefault('ready_at', time.monotonic() + s.get('delay', 0.1))
save()
'''


@unittest.skipUnless(shutil.which('zsh'), 'zsh is required for the prefetch script')
class PrefetchFlowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.fake = self.root / 'fake'
        self.fake.mkdir()
        (self.root / 'bin').mkdir()
        for name in ['find', 'stat', 'brctl']:
            p = self.fake / name
            p.write_text('#!' + sys.executable + '\n' + FAKE_OS)
            p.chmod(0o755)
        p = self.root / 'bin' / 'icloud-download'
        p.write_text('#!' + sys.executable + '\n' + FAKE_OS)
        p.chmod(0o755)
        # Replace OS boundaries in a copy, preserving the script's actual scheduler and locks.
        script = (EXAMPLE / 'icloud-prefetch.sh').read_text()
        for name in ['find', 'stat', 'brctl']:
            script = script.replace('/usr/bin/' + name, shlex.quote(str(self.fake / name)))
        self.script = self.root / 'icloud-prefetch.sh'
        self.script.write_text(script)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('SYNC_')}
        self.env.update(FAKE_SYNC_ROOT=str(self.root), SYNC_SOURCE=str(self.source),
                        PATH=str(self.fake) + os.pathsep + self.env['PATH'],
                        SYNC_DOWNLOAD_BATCH_SIZE='4', SYNC_DOWNLOAD_WAIT_SECONDS='3',
                        SYNC_DOWNLOAD_POLL_SECONDS='1', SYNC_DOWNLOAD_RETRY_SECONDS='10',
                        SYNC_DOWNLOAD_REQUEST_TIMEOUT_SECONDS='1', SYNC_READ_TIMEOUT_SECONDS='1')

    def scenario(self, files, **options):
        for name in files:
            p = self.source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'example')
        (self.root / 'scenario.json').write_text(json.dumps(dict(files=files, **options)))

    def run_script(self, expected=0):
        result = subprocess.run(['zsh', str(self.script)], env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr + self.log())
        self.assertFalse((self.root / 'state' / 'icloud_prefetch.lock.d').exists())
        return result

    def log(self):
        p = self.root / 'logs' / 'icloud-prefetch.log'
        return p.read_text() if p.exists() else ''

    def events(self):
        p = self.root / 'events.jsonl'
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

    def test_permission_fallback_is_selected_once_and_requests_are_batched(self):
        names = ['notes with spaces.md', '模型.py', 'line\nbreak.txt', 'dataless-title.md', 'last.txt']
        self.scenario(dict.fromkeys(names, {}), helper_denied=True)
        self.run_script()
        calls = self.events()
        self.assertEqual(sum(c[0] == 'icloud-download' for c in calls), 1)
        native = [c for c in calls if c[0] == 'brctl']
        self.assertEqual([c[1] for c in native], names)
        self.assertLess(native[3][2] - native[0][2], 0.9)
        self.assertIn('completed=5', self.log())
        self.assertEqual(self.log().count('helper unavailable'), 1)

    def test_helper_is_retried_on_a_later_round_after_permission_is_fixed(self):
        self.scenario({'first.md': {}}, helper_denied=True)
        self.run_script()
        self.scenario({'second.md': {}}, helper_denied=False)
        self.run_script()
        second = [e[0] for e in self.events() if e[1] == 'second.md']
        self.assertEqual(second, ['icloud-download'])

    def test_stuck_file_does_not_hold_the_window_and_is_not_reported_complete(self):
        self.env['SYNC_DOWNLOAD_BATCH_SIZE'] = '2'
        self.scenario({'slow.md': {'stuck': True}, 'fast.md': {}, 'later.md': {}})
        self.run_script(expected=1)
        events = self.events()
        first_time = events[0][2]
        self.assertLess(next(e[2] for e in events if e[1] == 'later.md') - first_time, 2.5)
        self.assertEqual(len(events), 3)  # no duplicate initial request
        self.assertIn('completed=2 timeouts=1 failures=0', self.log())
        self.assertIn('slow.md', (self.root / 'logs' / 'icloud-prefetch.last-errors.log').read_text())

    def test_request_command_has_a_deadline(self):
        self.scenario({'hang.md': {'request_hang': True}, 'ok.md': {}})
        began = time.monotonic()
        self.run_script(expected=1)
        self.assertLess(time.monotonic() - began, 8)
        self.assertIn('completed=1 timeouts=1', self.log())

    def test_disappearing_or_unreadable_file_is_not_counted_complete(self):
        self.scenario({'gone.md': {'vanish': True}, 'unreadable.md': {'stat_error': True}, 'ok.md': {}})
        self.run_script(expected=1)
        self.assertIn('completed=1 timeouts=0 failures=2', self.log())

    def test_scan_failure_and_invalid_settings_stop_before_download(self):
        self.scenario({'a.md': {}}, scan_error=True)
        self.run_script(expected=1)
        self.assertEqual(self.events(), [])
        for invalid in ['0', '00', '-1', 'text', '257']:
            self.env['SYNC_DOWNLOAD_BATCH_SIZE'] = invalid
            self.run_script(expected=2)
        self.assertEqual(self.events(), [])

    def test_dead_lock_and_per_round_limit(self):
        self.scenario({f'{i}.md': {} for i in range(5)})
        lock = self.root / 'state' / 'icloud_prefetch.lock.d'
        lock.mkdir(parents=True)
        (lock / 'pid').write_text('999999999')
        (lock / 'candidates').write_text('old queue')
        self.env['SYNC_PREFETCH_LIMIT'] = '2'
        self.run_script()
        self.assertEqual(len(self.events()), 2)
        self.assertIn('completed=2', self.log())

    def test_live_lock_and_termination_cleanup(self):
        self.scenario({'slow.md': {'stuck': True}})
        self.env['SYNC_DOWNLOAD_WAIT_SECONDS'] = '30'
        proc = subprocess.Popen(['zsh', str(self.script)], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not self.events() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(self.events())
            other = subprocess.run(['zsh', str(self.script)], env=self.env, capture_output=True, text=True, timeout=5)
            self.assertEqual(other.returncode, 0, other.stderr)
            self.assertEqual(len(self.events()), 1)
            proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 143)
            self.assertFalse((self.root / 'state' / 'icloud_prefetch.lock.d').exists())
        finally:
            if proc.poll() is None:
                proc.kill(); proc.communicate()

FAKE_PUSH = r'''
import json, os, pathlib, shutil, subprocess, sys, time
base = pathlib.Path(os.environ['FAKE_SYNC_ROOT'])
cmd = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with (base / 'push-events.jsonl').open('a') as out:
    out.write(json.dumps([cmd, args, time.time()]) + '\n')
if cmd == 'find':
    if '-flags' in args:
        if os.environ.get('FAKE_SCAN_ERROR'):
            print('scan failed', file=sys.stderr); sys.exit(1)
        if os.environ.get('FAKE_PLACEHOLDER'):
            print(str(base / 'source' / 'document.txt'))
        sys.exit(0)
    sys.exit(subprocess.run(['/usr/bin/find', *args]).returncode)
if cmd == 'ssh':
    command = args[-1]
    if command == 'true' and os.environ.get('FAKE_OFFLINE'): sys.exit(255)
    if 'touch ' in command and 'mirror_changed.flag' in command and os.environ.get('FAKE_FLAG_FAIL'):
        sys.exit(255)
    sys.exit(subprocess.run(['/bin/sh', '-c', command]).returncode)
if cmd == 'rsync':
    time.sleep(float(os.environ.get('FAKE_TRANSFER_SLEEP', '0')))
    if '--dry-run' not in args and not os.environ.get('FAKE_NO_CHANGES'):
        remote = pathlib.Path(args[-1].split(':', 1)[1])
        remote.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / 'source' / 'document.txt', remote / 'document.txt')
    if not os.environ.get('FAKE_NO_CHANGES'): print('>f+++++++++ document.txt')
    sys.exit(int(os.environ.get('FAKE_TRANSFER_EXIT', '0')))
'''


class PushFlowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'source').mkdir()
        (self.root / 'source' / 'document.txt').write_text('example')
        self.fake = self.root / 'fake'
        self.fake.mkdir()
        for name in ['ssh', 'rsync', 'find']:
            p = self.fake / name
            p.write_text('#!' + sys.executable + '\n' + FAKE_PUSH)
            p.chmod(0o755)
        source = (EXAMPLE / 'push-mirror.sh').read_text()
        source = source.replace('/usr/bin/find', shlex.quote(str(self.fake / 'find')))
        self.script = self.root / 'push-mirror.sh'
        self.script.write_text(source)
        self.remote = self.root / 'server'
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('SYNC_')}
        self.env.update(FAKE_SYNC_ROOT=str(self.root), SYNC_SOURCE=str(self.root / 'source'),
                        SYNC_HOST='test-server', SYNC_REMOTE_DIR=str(self.remote),
                        SYNC_REMOTE_LOCK_STALE_SECONDS='3', PATH=str(self.fake) + os.pathsep + self.env['PATH'])

    def events(self):
        p = self.root / 'push-events.jsonl'
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

    def run_script(self, *args, expected=0):
        p = subprocess.run(['bash', str(self.script), '--allow-no-launchd', *args],
                           env=self.env, capture_output=True, text=True, timeout=15)
        log = self.root / 'logs' / 'push-mirror.log'
        self.assertEqual(p.returncode, expected, p.stderr + (log.read_text() if log.exists() else ''))
        self.assertFalse((self.root / 'state' / 'mirror_push.lock.d').exists())
        return p

    def test_dry_run_makes_no_server_changes(self):
        self.run_script('--dry-run')
        self.assertFalse(self.remote.exists())
        ssh_commands = [e[1][-1] for e in self.events() if e[0] == 'ssh']
        self.assertEqual(ssh_commands, ['true'])
        self.assertFalse((self.root / 'state' / 'mirror_flag_pending').exists())

    def test_pending_download_and_scan_failure_prevent_transfer(self):
        self.env['FAKE_PLACEHOLDER'] = '1'
        self.run_script()
        self.assertFalse(any(e[0] in ['ssh', 'rsync'] for e in self.events()))
        self.env.pop('FAKE_PLACEHOLDER')
        self.env['FAKE_SCAN_ERROR'] = '1'
        self.run_script()
        self.assertFalse(any(e[0] in ['ssh', 'rsync'] for e in self.events()))

    def test_prefetch_lock_and_offline_server_defer_transfer(self):
        lock = self.root / 'state' / 'icloud_prefetch.lock.d'
        lock.mkdir(parents=True)
        self.run_script()
        self.assertFalse(any(e[0] == 'rsync' for e in self.events()))
        lock.rmdir()
        self.env['FAKE_OFFLINE'] = '1'
        self.run_script()
        self.assertFalse(any(e[0] == 'rsync' for e in self.events()))

    def test_success_heartbeat_and_failed_transfer_notify_server(self):
        self.env['FAKE_TRANSFER_SLEEP'] = '2.3'
        self.run_script()
        state = self.remote / 'runtime' / 'state'
        self.assertTrue((state / 'mirror_changed.flag').exists())
        self.assertFalse((state / 'mirror_sync.lock.d').exists())
        self.assertTrue(any(e[0] == 'ssh' and '&& touch ' in e[1][-1] for e in self.events()))
        self.assertFalse((self.root / 'state' / 'mirror_flag_pending').exists())
        (state / 'mirror_changed.flag').unlink()
        self.env['FAKE_TRANSFER_SLEEP'] = '0'
        self.env['FAKE_TRANSFER_EXIT'] = '1'
        self.run_script(expected=1)
        self.assertTrue((state / 'mirror_changed.flag').exists())
        self.assertFalse((state / 'mirror_sync.lock.d').exists())

    def test_failed_notification_is_retried_after_a_noop(self):
        self.env['FAKE_FLAG_FAIL'] = '1'
        self.run_script()
        pending = self.root / 'state' / 'mirror_flag_pending'
        self.assertTrue(pending.exists())
        self.env['FAKE_NO_CHANGES'] = '1'
        self.run_script()
        self.assertTrue(pending.exists())
        self.env.pop('FAKE_FLAG_FAIL')
        self.run_script()
        self.assertFalse(pending.exists())
        flag = self.remote / 'runtime' / 'state' / 'mirror_changed.flag'
        self.assertTrue(flag.exists())
        flag.unlink()
        self.run_script()
        self.assertFalse(flag.exists())  # ordinary no-op rounds do not re-arm the scanner
        self.assertFalse(pending.exists())

    def test_killed_transfer_retains_notification_intent(self):
        self.env['FAKE_TRANSFER_SLEEP'] = '1.2'
        proc = subprocess.Popen(['bash', str(self.script), '--allow-no-launchd'], env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not any(e[0] == 'rsync' for e in self.events()) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(any(e[0] == 'rsync' for e in self.events()))
            self.assertTrue((self.root / 'state' / 'mirror_flag_pending').exists())
            proc.kill()
            proc.communicate(timeout=5)  # fake transfer and heartbeat finish after the parent dies
            self.env['FAKE_TRANSFER_SLEEP'] = '0'
            self.env['FAKE_NO_CHANGES'] = '1'
            remote_lock = self.remote / 'runtime' / 'state' / 'mirror_sync.lock.d'
            os.utime(remote_lock, (0, 0))
            self.run_script()
            self.assertTrue((self.remote / 'runtime' / 'state' / 'mirror_changed.flag').exists())
            self.assertFalse(remote_lock.exists())
        finally:
            if proc.poll() is None:
                proc.kill(); proc.communicate()


class InstallFlowTests(unittest.TestCase):
    def test_unload_waits_until_launchd_has_removed_the_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'launchctl'
            fake.write_text('#!' + sys.executable + '\n' + r'''
from pathlib import Path
import os, sys
p = Path(os.environ['FAKE_LAUNCH_STATE'])
with p.with_suffix('.calls').open('a') as out: out.write(sys.argv[1] + '\n')
if sys.argv[1] == 'bootout': p.write_text('2'); sys.exit(0)
left = int(p.read_text())
if left:
    p.write_text(str(left - 1)); sys.exit(0)
sys.exit(1)
''')
            fake.chmod(0o755)
            source = (EXAMPLE / 'install.sh').read_text()
            function = source[source.index('unload() {'):source.index('\nif [[ "$mode" == uninstall')]
            env = os.environ.copy()
            env.update(FAKE_LAUNCH_STATE=str(root / 'state'), PATH=str(root) + os.pathsep + env['PATH'])
            p = subprocess.run(['bash', '-c', 'DOMAIN=gui/test\n' + function + '\nunload test-agent'],
                               env=env, capture_output=True, text=True, timeout=8)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual((root / 'state.calls').read_text().splitlines(), ['bootout', 'print', 'print', 'print'])
