import Link from 'next/link';

interface ProjectCardProps {
  title: string;
  description?: string;
  imageUrl?: string;
  location?: string;
  viewerUrl?: string;
}

const ProjectCard = ({ title, description, imageUrl, location, viewerUrl }: ProjectCardProps) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
      <div className="h-48 bg-gray-100 flex items-center justify-center">
        {imageUrl ? (
          <img src={imageUrl} alt={title} className="w-full h-full object-cover" />
        ) : (
          <div className="text-gray-400">
            <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
        )}
      </div>
      <div className="p-4">
        <h3 className="text-lg font-semibold text-black mb-2">{title}</h3>
        {location && (
          <p className="text-sm text-gray-600 mb-2">{location}</p>
        )}
        {description && (
          <p className="text-sm text-gray-700 mb-4">{description}</p>
        )}
        <div className="flex gap-2">
          {viewerUrl && (
            <Link
              href={viewerUrl}
              className="text-sm px-4 py-2 bg-blue-400 text-white rounded hover:bg-blue-500 transition-colors"
            >
              View 3D Model
            </Link>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectCard;

