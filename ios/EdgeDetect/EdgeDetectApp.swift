import SwiftUI

@main
struct EdgeDetectApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                LiveView()
                    .tabItem { Label("Live", systemImage: "camera.viewfinder") }
                BenchmarkView()
                    .tabItem { Label("Benchmark", systemImage: "speedometer") }
            }
        }
    }
}
