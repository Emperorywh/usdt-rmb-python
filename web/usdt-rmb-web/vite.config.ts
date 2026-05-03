import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// ============================================================
// Vite 配置
// ------------------------------------------------------------
// 1) tailwindcss() —— Tailwind v4 官方插件，自动扫描 src 下的类名；
// 2) react()       —— React Fast Refresh；
// 3) server.proxy  —— 把前端 /api/* 转发到 FastAPI（默认 :8000），
//    后端路由是 /analysis/latest 等不带前缀，所以需要 rewrite 去掉 /api。
//    这样前端代码里统一写 fetch('/api/analysis/latest') 即可。
// ============================================================
export default defineConfig({
  plugins: [tailwindcss(), react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
