# MemoryLayer Admin Dashboard

Web administration for MemoryLayer deployments, built with Next.js, React,
Tailwind CSS, TanStack Query, and the Apache-2.0 MemoryLayer TypeScript SDK.

Manage workspaces, memories, API tokens, sessions, documents, datasets, jobs,
audit logs, retrieval trajectories, entities, and storage tiering.

## Development

Requires Node.js 22+ and the matching MemoryLayer SDK 0.2.0. See
[dependency availability](../docs/DEVELOPMENT.md); the public 0.1.22 SDK does
not provide the required API surface.

```sh
npm install
npm run typecheck
npm run build
npm run dev
```

Open `http://localhost:3200`. Set `MEMORYLAYER_URL` to your backend URL before
building/running Next.js; `/api/ml` forwards requests there. Configure `/api/ml`
and your admin-scoped API key in Settings. Keep API keys out of source files
and `NEXT_PUBLIC_*` variables.

The application remains `private: true` in package.json to prevent accidental
npm publication; this does not restrict the source license.

## License

Scitrera's dashboard extensions are [AGPL-3.0-only](LICENSE).
Adapted shadcn/ui components retain their MIT attribution in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The MemoryLayer SDK retains
Apache-2.0.
