import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { useNavigate } from "react-router";
import { Upload, Play, Camera, LogOut } from "lucide-react";
import logoImage from "figma:asset/bab8d582dc7cf09d6de4400a2da88a55b3bd630c.png";
import techBgImage from "figma:asset/c27a4053dbdd89fddd476c8bc6fdc408749c89cb.png";

export function DashboardPage() {
  const navigate = useNavigate();

  const handleFileSelect = () => {
    // 选择文件逻辑
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,video/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        console.log('Selected file:', file.name);
      }
    };
    input.click();
  };

  const handleStartRecognition = () => {
    // 开始识别逻辑
    alert('开始识别...');
  };

  const handleOpenCamera = () => {
    // 打开摄像头逻辑
    alert('打开摄像头...');
  };

  const handleExit = () => {
    // 退出系统
    if (confirm('确定要退出系统吗？')) {
      navigate('/');
    }
  };

  return (
    <div 
      className="min-h-screen relative overflow-hidden"
      style={{
        backgroundImage: `url(${techBgImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      {/* 技术背景叠加层 */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/20 via-blue-400/20 to-blue-600/30"></div>
      
      {/* 白色网格线叠加效果 */}
      <div 
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px)
          `,
          backgroundSize: "50px 50px"
        }}
      ></div>

      {/* 顶部Logo和标题区域 */}
      <div className="relative z-10 pt-8 pr-8 flex justify-end items-center gap-4">
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center shadow-2xl">
          <img src={logoImage} alt="DriveSafe Logo" className="w-full h-full object-cover rounded-full" />
        </div>
        <div className="text-right">
          <h1 className="text-2xl font-serif italic text-blue-900">DriveSafe</h1>
          <div className="h-1 w-full bg-gradient-to-r from-transparent via-blue-600 to-blue-400 mt-1"></div>
        </div>
      </div>

      {/* 内容区域 */}
      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* 顶部横向卡片组 - 7:3比例居中 */}
        <div className="flex justify-center gap-8 mb-8">
          {/* 左侧主要内容区域 - 70% */}
          <Card className="bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl p-8 h-[400px] border-4 border-white/50 flex-[7]">
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">📊</div>
                <p className="text-lg">主要内容区域</p>
              </div>
            </div>
          </Card>

          {/* 右侧数据卡片 - 30% */}
          <Card className="bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl p-6 h-[400px] border-4 border-white/50 flex-[3]">
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-4xl mb-2">📈</div>
                <p>数据卡片</p>
              </div>
            </div>
          </Card>
        </div>

        {/* 底部区域 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* 底部横向卡片 - 车辆监控信息 */}
          <Card className="lg:col-span-1 bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl p-6 h-[150px] border-4 border-white/50">
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-4xl mb-2">🚗</div>
                <p className="text-base">车辆监控信息</p>
              </div>
            </div>
          </Card>

          {/* 功能按钮区域 - 右下角 */}
          <div className="lg:col-span-1 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Button
                onClick={handleFileSelect}
                className="h-16 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-2xl shadow-xl flex items-center justify-center gap-2"
              >
                <Upload className="w-5 h-5" />
                <span>选择文件</span>
              </Button>
              
              <Button
                onClick={handleStartRecognition}
                className="h-16 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white rounded-2xl shadow-xl flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                <span>开始识别</span>
              </Button>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <Button
                onClick={handleOpenCamera}
                className="h-16 bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-600 hover:to-cyan-700 text-white rounded-2xl shadow-xl flex items-center justify-center gap-2"
              >
                <Camera className="w-5 h-5" />
                <span>打开摄像头</span>
              </Button>
              
              <Button
                onClick={handleExit}
                className="h-16 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-2xl shadow-xl flex items-center justify-center gap-2"
              >
                <LogOut className="w-5 h-5" />
                <span>退出系统</span>
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* 装饰性科技元素 */}
      <div className="absolute top-1/4 left-1/4 w-64 h-64 border-2 border-cyan-300/30 rounded-full animate-pulse-ring"></div>
      <div className="absolute bottom-1/4 right-1/4 w-48 h-48 border-2 border-blue-300/30 rounded-full animate-pulse-ring" style={{ animationDelay: '1s' }}></div>
      <div className="absolute top-1/2 right-1/3 w-32 h-32 border-2 border-cyan-400/20 rounded-full animate-pulse-ring" style={{ animationDelay: '2s' }}></div>
    </div>
  );
}