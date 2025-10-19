import Link from "next/link";
import Navigation from "@/components/Navigation";
import ProjectCard from "@/components/ProjectCard";

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-black mb-6">
            UT3C Heritage
          </h1>
          <p className="text-xl text-gray-700 mb-8 max-w-3xl mx-auto">
            A collaborative platform for collecting, sharing, and curating 3D captures
            of heritage places and spatial designs. Preservation through digital innovation.
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              href="/viewer"
              className="px-6 py-3 bg-blue-400 text-white rounded-lg hover:bg-blue-500 transition-colors font-medium"
            >
              Explore 3D Viewer
            </Link>
            <Link
              href="/projects"
              className="px-6 py-3 border-2 border-blue-400 text-blue-400 rounded-lg hover:bg-blue-50 transition-colors font-medium"
            >
              View Projects
            </Link>
          </div>
        </div>
      </section>

      {/* Mission Statement */}
      <section className="bg-blue-50 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-black mb-4">
              Digital Models and Stories
            </h2>
            <p className="text-lg text-gray-700">
              A gallery of individually collected 3D captures and collaboratively compiled narratives
              about heritage places and their spatial designs. Our mission is to preserve cultural
              heritage through cutting-edge 3D digitization technology.
            </p>
          </div>
        </div>
      </section>

      {/* Featured Projects */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl font-bold text-black mb-8">Featured Projects</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <ProjectCard
            title="Auditorio Example"
            location="Sample Location"
            description="A demonstration of our 3D heritage preservation capabilities using advanced PLY visualization."
            viewerUrl="/viewer"
          />
          <ProjectCard
            title="Heritage Site Documentation"
            location="TBD"
            description="Comprehensive digital documentation of historic architecture and cultural landmarks."
          />
          <ProjectCard
            title="Cultural Preservation Initiative"
            location="TBD"
            description="Collaborative project to preserve and share endangered cultural heritage sites."
          />
        </div>
        <div className="text-center mt-12">
          <Link
            href="/projects"
            className="inline-block px-6 py-3 border-2 border-blue-400 text-blue-400 rounded-lg hover:bg-blue-50 transition-colors font-medium"
          >
            View All Projects
          </Link>
        </div>
      </section>

      {/* Technology Section */}
      <section className="bg-gray-50 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-black mb-12 text-center">Our Technology</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-400 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-black mb-2">3D Scanning</h3>
              <p className="text-gray-700">
                High-precision 3D scanning technology for accurate heritage site documentation
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-400 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-black mb-2">Interactive Visualization</h3>
              <p className="text-gray-700">
                Immersive 3D viewer with advanced controls for exploring heritage sites
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-400 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-black mb-2">Digital Preservation</h3>
              <p className="text-gray-700">
                Long-term digital archiving and preservation of cultural heritage data
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-gray-600">
            <p className="mb-2">UT3C Heritage - Digital Heritage Preservation Platform</p>
            <p className="text-sm">Preserving the past, building the future through technology</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
