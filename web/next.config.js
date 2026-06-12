/** @type {import('next').NextConfig} */
const nextConfig = {
  // API proxy: forward /api/* requests to the FastAPI backend during development
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
