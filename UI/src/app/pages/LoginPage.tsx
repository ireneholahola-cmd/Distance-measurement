import { useState } from "react";
import { useNavigate } from "react-router";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import logoImage from "figma:asset/bab8d582dc7cf09d6de4400a2da88a55b3bd630c.png";
import bgImage from "figma:asset/322312e4a0bbf489528e255e522c7cc1041d5add.png";

export function LoginPage() {
  const navigate = useNavigate();
  const [account, setAccount] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = () => {
    // 模拟登录逻辑
    if (account && password) {
      navigate("/dashboard");
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center relative overflow-hidden"
      style={{
        backgroundImage: `url(${bgImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      {/* 深色半透明遮罩层 */}
      <div className="absolute inset-0 bg-black/30"></div>
      
      {/* 渐变叠加层 */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/15 via-blue-500/15 to-blue-900/25"></div>
      
      {/* 柔和光晕效果 */}
      <div className="absolute inset-0 bg-gradient-to-t from-white/5 via-transparent to-white/10"></div>
      
      {/* 左侧品牌区域 */}
      <div className="absolute left-12 top-1/2 -translate-y-1/2 z-10 hidden lg:block">
        <div className="flex flex-col items-start gap-4">
          <div className="w-48 h-48 rounded-full bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center shadow-2xl">
            <img src={logoImage} alt="DriveSafe Logo" className="w-full h-full object-cover rounded-full" />
          </div>
          <div className="text-left">
            <h1 className="text-4xl font-bold text-white drop-shadow-lg mb-2">驭安</h1>
            <h2 className="text-3xl font-serif italic text-cyan-300 drop-shadow-md">DriveSafe</h2>
          </div>
        </div>
      </div>

      {/* 登录表单容器 */}
      <div className="relative z-10 w-full max-w-md mx-auto px-4">
        <div className="bg-white/95 backdrop-blur-sm rounded-3xl shadow-2xl p-12 border border-white/20">
          {/* 移动端Logo */}
          <div className="flex flex-col items-center mb-8 lg:hidden">
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center shadow-xl mb-4">
              <img src={logoImage} alt="DriveSafe Logo" className="w-full h-full object-cover rounded-full" />
            </div>
            <div className="text-center">
              <h1 className="text-2xl font-bold text-slate-900">驭安</h1>
              <h2 className="text-xl font-serif italic text-blue-600">DriveSafe</h2>
            </div>
          </div>

          {/* 欢迎标题 */}
          <h2 className="text-3xl text-center font-medium text-blue-600 mb-8 border-b-2 border-blue-400 pb-3">
            欢迎登录
          </h2>

          {/* 输入框 */}
          <div className="space-y-6 mb-6">
            <Input
              type="text"
              placeholder="请输入账号"
              value={account}
              onChange={(e) => setAccount(e.target.value)}
              className="h-14 bg-white border-2 border-gray-200 rounded-xl text-base px-4 focus:border-blue-400 focus:ring-2 focus:ring-blue-200"
            />
            <Input
              type="password"
              placeholder="请输入密码"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="h-14 bg-white border-2 border-gray-200 rounded-xl text-base px-4 focus:border-blue-400 focus:ring-2 focus:ring-blue-200"
            />
          </div>

          {/* 忘记密码 */}
          <div className="text-right mb-6">
            <a href="#" className="text-sm text-blue-500 hover:text-blue-700">
              忘记密码?
            </a>
          </div>

          {/* 登录按钮 */}
          <Button
            onClick={handleLogin}
            className="w-full h-14 bg-gradient-to-r from-cyan-400 to-blue-500 hover:from-cyan-500 hover:to-blue-600 text-white text-lg font-medium rounded-xl shadow-lg"
          >
            登录
          </Button>

          {/* 底部链接 */}
          <div className="flex justify-between items-center mt-6 text-sm text-gray-600">
            <a href="#" className="hover:text-blue-600">
              其他账号登陆
            </a>
            <button 
              onClick={() => navigate("/register")}
              className="hover:text-blue-600"
            >
              立即注册
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}