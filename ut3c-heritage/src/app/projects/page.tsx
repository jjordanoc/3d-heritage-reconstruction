import Navigation from "@/components/Navigation";
import ProjectCard from "@/components/ProjectCard";

export default function ProjectsPage() {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Page Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-black mb-4">
            Digital Models and Stories
          </h1>
          <p className="text-lg text-gray-700 max-w-3xl">
            A comprehensive gallery of individually collected 3D captures and collaboratively
            compiled narratives about heritage places and their spatial designs.
          </p>
        </div>

        {/* Featured Projects Section */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-black mb-6 pb-2 border-b-2 border-blue-400">
            Featured Projects
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <ProjectCard
              title="Auditorio Example"
              location="Sample Location"
              description="A demonstration of our 3D heritage preservation capabilities using advanced PLY visualization technology."
              viewerUrl="/viewer"
            />
            <ProjectCard
              title="Historic Architecture Documentation"
              location="TBD"
              description="Comprehensive digital documentation of historic architectural landmarks and their cultural significance."
            />
            <ProjectCard
              title="Cultural Heritage Sites"
              location="TBD"
              description="Preservation of endangered cultural heritage sites through 3D digitization and interactive exploration."
            />
          </div>
        </section>

        {/* All Projects Section */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-black mb-6 pb-2 border-b-2 border-blue-400">
            All Projects
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <ProjectCard
              title="Religious Architecture Study"
              location="Various Locations"
              description="A comparative study of religious architectural heritage across different cultures and time periods."
            />
            <ProjectCard
              title="Urban Heritage Mapping"
              location="City Centers"
              description="Mapping and preserving urban heritage sites before modern development changes the landscape."
            />
            <ProjectCard
              title="Traditional Craftsmanship"
              location="Workshops"
              description="Documenting traditional craftsmanship spaces and techniques for future generations."
            />
            <ProjectCard
              title="Archaeological Sites"
              location="Field Research"
              description="3D documentation of archaeological excavations and discovered artifacts."
            />
            <ProjectCard
              title="Historic Districts"
              location="Old Towns"
              description="Complete digital preservation of historic districts with architectural and cultural significance."
            />
            <ProjectCard
              title="Monument Restoration"
              location="Various Sites"
              description="Before and after documentation of heritage monument restoration projects."
            />
          </div>
        </section>

        {/* Call to Action */}
        <section className="bg-blue-50 rounded-lg p-8 text-center">
          <h3 className="text-2xl font-bold text-black mb-4">
            Want to Contribute?
          </h3>
          <p className="text-gray-700 mb-6 max-w-2xl mx-auto">
            UT3C Heritage welcomes collaborations from researchers, institutions, and
            cultural heritage enthusiasts. Help us preserve our shared heritage through
            digital innovation.
          </p>
          <button className="px-6 py-3 bg-blue-400 text-white rounded-lg hover:bg-blue-500 transition-colors font-medium">
            Get Involved
          </button>
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

