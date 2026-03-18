import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

// 自定义插件来处理figma:asset/导入
const figmaAssetPlugin = () => {
  return {
    name: 'figma-asset-resolver',
    enforce: 'pre',
    resolveId(source, importer) {
      // 处理figma:asset/前缀的导入
      if (source.startsWith('figma:asset/')) {
        // 提取文件名
        const filename = source.replace('figma:asset/', '')
        // 转换为相对于src/assets的路径
        return path.resolve(__dirname, './src/assets', filename)
      }
      return null
    },
  }
}

export default defineConfig({
  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
    figmaAssetPlugin(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
})
