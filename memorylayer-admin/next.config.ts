import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/ml/:path*",
        destination: `${process.env.MEMORYLAYER_URL || "http://ml-aether-auth-proxy:8080"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
