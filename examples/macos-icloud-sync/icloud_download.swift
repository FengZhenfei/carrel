// Ask iCloud Drive to download the given files: a placeholder becomes a real file once the download finishes.
// Prints one line per file; exits 1 if any request failed. icloud-prefetch.sh uses it as bin/icloud-download
// when present (install.sh builds it with swiftc from the Xcode Command Line Tools) and falls back to
// `brctl download` and a read probe otherwise.
import Foundation

let fm = FileManager.default
var exitCode: Int32 = 0

for arg in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: arg)
    do {
        try fm.startDownloadingUbiquitousItem(at: url)
        print("requested\t\(arg)")
    } catch {
        fputs("failed\t\(arg)\t\(error)\n", stderr)
        exitCode = 1
    }
}

exit(exitCode)
