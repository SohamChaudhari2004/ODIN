/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_DEV_MODE: string;
  readonly VITE_ENABLE_MOCK_DATA: string;
  readonly VITE_ODIN_SYSTEM_NAME: string;
  readonly VITE_ODIN_VERSION: string;
  readonly VITE_DEFAULT_SIMULATION_SPEED: string;
  readonly VITE_MAX_SIMULATION_SPEED: string;
  readonly VITE_API_TIMEOUT: string;
  readonly VITE_WS_RECONNECT_INTERVAL: string;
  readonly VITE_LOG_LEVEL: string;
  readonly VITE_ENABLE_CONSOLE_LOGS: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
