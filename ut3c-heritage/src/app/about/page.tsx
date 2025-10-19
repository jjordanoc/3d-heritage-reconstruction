import Navigation from "@/components/Navigation";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Page Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-black mb-6">About UT3C Heritage</h1>
          <div className="prose prose-lg max-w-none">
            <p className="text-lg text-gray-700 mb-4">
              UT3C Heritage is a collaborative platform for collecting, sharing, and curating
              3D captures of heritage places and spatial designs. We provide an online tool
              for preservation, education, and community development.
            </p>
            <p className="text-lg text-gray-700 mb-4">
              Our mission is to preserve cultural heritage through cutting-edge technology,
              making it accessible to researchers, educators, and the general public worldwide.
              Through high-precision 3D scanning and interactive visualization, we create
              digital twins of heritage sites that can be explored, studied, and preserved
              for future generations.
            </p>
          </div>
        </div>

        {/* Mission & Vision */}
        <section className="mb-16">
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-blue-50 p-8 rounded-lg">
              <h2 className="text-2xl font-bold text-black mb-4">Our Mission</h2>
              <p className="text-gray-700">
                To preserve and share cultural heritage through advanced 3D digitization
                technology, making it accessible to everyone while supporting research,
                education, and cultural understanding across the globe.
              </p>
            </div>
            <div className="bg-gray-50 p-8 rounded-lg">
              <h2 className="text-2xl font-bold text-black mb-4">Our Vision</h2>
              <p className="text-gray-700">
                A world where cultural heritage is digitally preserved and universally
                accessible, fostering appreciation and understanding of diverse cultures
                and histories through innovative technology.
              </p>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-black mb-8 pb-2 border-b-2 border-blue-400">
            Our Team
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-black mb-2">Project Director</h3>
              <p className="text-gray-600 mb-4">Leading the vision and strategy</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-black mb-2">Development Lead</h3>
              <p className="text-gray-600 mb-4">Overseeing technical implementation</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-black mb-2">Research Team</h3>
              <p className="text-gray-600 mb-4">Conducting heritage documentation</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-black mb-2">3D Specialists</h3>
              <p className="text-gray-600 mb-4">Experts in 3D scanning and modeling</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-black mb-2">Conservation Experts</h3>
              <p className="text-gray-600 mb-4">Heritage preservation specialists</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-black mb-2">Community Partners</h3>
              <p className="text-gray-600 mb-4">Collaborating with local communities</p>
            </div>
          </div>
        </section>

        {/* Technology */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-black mb-8 pb-2 border-b-2 border-blue-400">
            Our Technology
          </h2>
          <div className="space-y-6">
            <div className="flex gap-6 items-start">
              <div className="w-12 h-12 bg-blue-400 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-black mb-2">Advanced 3D Scanning</h3>
                <p className="text-gray-700">
                  We utilize state-of-the-art 3D scanning technology including LiDAR and
                  photogrammetry to capture heritage sites with millimeter-level precision.
                </p>
              </div>
            </div>
            <div className="flex gap-6 items-start">
              <div className="w-12 h-12 bg-blue-400 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-black mb-2">Interactive Visualization</h3>
                <p className="text-gray-700">
                  Our custom-built PLY viewer powered by Three.js allows users to explore
                  heritage sites in full 3D with intuitive controls and high-quality rendering.
                </p>
              </div>
            </div>
            <div className="flex gap-6 items-start">
              <div className="w-12 h-12 bg-blue-400 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-black mb-2">Digital Preservation</h3>
                <p className="text-gray-700">
                  We implement robust digital preservation strategies to ensure long-term
                  accessibility and integrity of cultural heritage data for future generations.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="bg-blue-50 rounded-lg p-8 text-center">
          <h3 className="text-2xl font-bold text-black mb-4">
            Join Our Mission
          </h3>
          <p className="text-gray-700 mb-6 max-w-2xl mx-auto">
            Whether you're a researcher, institution, or cultural heritage enthusiast,
            we welcome collaborations to help preserve our shared heritage.
          </p>
          <div className="flex gap-4 justify-center">
            <button className="px-6 py-3 bg-blue-400 text-white rounded-lg hover:bg-blue-500 transition-colors font-medium">
              Contact Us
            </button>
            <button className="px-6 py-3 border-2 border-blue-400 text-blue-400 rounded-lg hover:bg-blue-50 transition-colors font-medium">
              Learn More
            </button>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8 mt-16">
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

