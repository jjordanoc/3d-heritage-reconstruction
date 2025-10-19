"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const Navigation = () => {
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <nav className="w-full bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="text-2xl font-bold text-black hover:text-blue-400 transition-colors">
            UT3C Heritage
          </Link>

          <div className="flex gap-8">
            <Link
              href="/"
              className={`text-sm font-medium transition-colors ${isActive('/')
                  ? 'text-blue-400'
                  : 'text-black hover:text-blue-400'
                }`}
            >
              Home
            </Link>
            <Link
              href="/projects"
              className={`text-sm font-medium transition-colors ${isActive('/projects')
                  ? 'text-blue-400'
                  : 'text-black hover:text-blue-400'
                }`}
            >
              Projects
            </Link>
            <Link
              href="/viewer"
              className={`text-sm font-medium transition-colors ${isActive('/viewer')
                  ? 'text-blue-400'
                  : 'text-black hover:text-blue-400'
                }`}
            >
              3D Viewer
            </Link>
            <Link
              href="/about"
              className={`text-sm font-medium transition-colors ${isActive('/about')
                  ? 'text-blue-400'
                  : 'text-black hover:text-blue-400'
                }`}
            >
              About
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;

