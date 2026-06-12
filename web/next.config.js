/** @type {import('next').NextConfig} */
const isGithubPages = process.env.DEPLOY_TARGET === "github-pages";

const nextConfig = {
  // Static export for GitHub Pages; dev mode uses rewrites to proxy the API
  ...(isGithubPages
    ? {
        output: "export",
        basePath: "/Gurbani-Guidance",
        assetPrefix: "/Gurbani-Guidance",
        images: { unoptimized: true },
      }
    : {
        async rewrites() {
          return [
            {
              source: "/api/:path*",
              destination: "http://localhost:8000/:path*",
            },
          ];
        },
      }),
};

module.exports = nextConfig;
