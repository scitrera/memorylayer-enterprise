import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

interface ConnectionConfig {
  baseUrl: string;
  apiKey?: string;
}

let clientInstance: MemoryLayerClient | null = null;
let currentConfig: ConnectionConfig | null = null;

export function getClient(config: ConnectionConfig): MemoryLayerClient {
  if (
    !clientInstance ||
    !currentConfig ||
    currentConfig.baseUrl !== config.baseUrl ||
    currentConfig.apiKey !== config.apiKey
  ) {
    clientInstance = new MemoryLayerClient({
      baseUrl: config.baseUrl,
      apiKey: config.apiKey,
      workspaceId: "_system",
      timeout: 30000,
    });
    currentConfig = { ...config };
  }
  return clientInstance;
}

export function getWorkspaceClient(config: ConnectionConfig, workspaceId: string): MemoryLayerClient {
  return new MemoryLayerClient({
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
    workspaceId,
    timeout: 30000,
  });
}

export function resetClient(): void {
  clientInstance = null;
  currentConfig = null;
}
