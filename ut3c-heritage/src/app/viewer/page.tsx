"use client";

import { useState } from 'react';
import PLYViewer from '@/components/PLYViewer';
import Navigation from '@/components/Navigation';
import ImageUpload from '@/components/ImageUpload';

const ViewerPage = () => {
  const projectId = 'auditorio';
  const [plyUrl, setPlyUrl] = useState(
    `https://huaca-pucllana--ut3c-heritage-backend-fastapi-app.modal.run/pointcloud/${projectId}/latest?t=${Date.now()}`
  );

  const handleUploadSuccess = () => {
    // Update PLY URL with new timestamp to force reload
    setPlyUrl(
      `https://huaca-pucllana--ut3c-heritage-backend-fastapi-app.modal.run/pointcloud/${projectId}/latest?t=${Date.now()}`
    );
  };

  return (
    <div className="h-screen flex flex-col bg-white">
      <Navigation />
      <div className="flex-1 flex flex-col">
        <div className="flex-1">
          <PLYViewer plyUrl={plyUrl} />
        </div>
        <ImageUpload projectId={projectId} onUploadSuccess={handleUploadSuccess} />
      </div>
    </div>
  );
};

export default ViewerPage;
