import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    Globe, Shield, Brain, Target, BarChart2,
    Users, Clock, LineChart, Database, Lightbulb 
} from 'lucide-react';

const GradientBackground = () => (
    <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-emerald-50 to-teal-100" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(34,197,94,0.2),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.25),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(52,211,153,0.2),transparent_60%)]" />
        <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-white/40 to-transparent" />
        <div className="absolute inset-0 bg-[linear-gradient(45deg,rgba(34,197,94,0.1)_25%,transparent_25%,transparent_50%,rgba(34,197,94,0.1)_50%,rgba(34,197,94,0.1)_75%,transparent_75%,transparent)] bg-[length:64px_64px] opacity-50" />
    </div>
);

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen relative overflow-hidden">
            <GradientBackground />
            
            <main className="container mx-auto px-4 py-12 relative">
                {/* Hero Section */}
                <section id="home" className="text-center mb-20 pt-16 relative">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(34,197,94,0.2),transparent_70%)]" />
                    <div className="relative">
                        <div className="inline-block">
                            <div className="relative">
                                <h2 className="text-6xl font-extrabold bg-gradient-to-r from-green-900 via-emerald-700 to-teal-900 bg-clip-text text-transparent mb-6 relative z-10">
                                    Shape Your Business Future with AI
                                </h2>
                                <div className="absolute -inset-1 bg-gradient-to-r from-green-500/30 via-emerald-500/30 to-teal-500/30 blur-3xl" />
                            </div>
                        </div>
                        <p className="text-xl text-green-900 max-w-3xl mx-auto mb-8 leading-relaxed backdrop-blur-sm bg-white/40 p-6 rounded-2xl shadow-lg border border-green-200/50">
                            Experience the power of MetaCorp's AI-driven business simulator that creates
                            parallel business realities, helping you foresee risks, identify opportunities,
                            and optimize your growth strategies with unprecedented accuracy.
                        </p>
                    </div>
                </section>

                {/* Features Grid */}
                <section className="grid md:grid-cols-3 gap-8 mb-20">
                    {[
                        {
                            Icon: Globe,
                            title: "Multiverse Simulation",
                            description: "Generate multiple future scenarios simultaneously, showing how decisions ripple across time and departments.",
                            gradient: "from-green-600 via-emerald-500 to-teal-600"
                        },
                        {
                            Icon: Brain,
                            title: "AI-Powered Insights",
                            description: "Leverage advanced AI to forecast trends, identify patterns, and uncover hidden opportunities in your data.",
                            gradient: "from-teal-600 via-green-500 to-emerald-600"
                        },
                        {
                            Icon: Shield,
                            title: "Risk Detection",
                            description: "Real-time anomaly detection and risk assessment to protect your business from potential threats.",
                            gradient: "from-emerald-600 via-teal-500 to-green-600"
                        },
                        {
                            Icon: LineChart,
                            title: "Advanced Analytics",
                            description: "Deep dive into your data with sophisticated analytical tools and predictive modeling.",
                            gradient: "from-green-600 via-teal-500 to-emerald-600"
                        },
                        {
                            Icon: Database,
                            title: "Data Integration",
                            description: "Seamlessly integrate with your existing data sources for comprehensive analysis.",
                            gradient: "from-teal-600 via-emerald-500 to-green-600"
                        },
                        {
                            Icon: Users,
                            title: "Team Collaboration",
                            description: "Enable team-wide decision making with shared insights and collaborative analysis.",
                            gradient: "from-emerald-600 via-green-500 to-teal-600"
                        }
                    ].map(({ Icon, title, description, gradient }, index) => (
                        <div key={index} className="group relative">
                            <div className="absolute inset-0 bg-gradient-to-br from-green-400/30 via-emerald-400/30 to-teal-400/30 rounded-2xl blur-2xl transform group-hover:scale-110 transition-transform duration-500" />
                            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(34,197,94,0.3),transparent_70%)] rounded-2xl" />
                            <div className="relative bg-gradient-to-br from-white/80 to-white/40 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <div className={`absolute top-0 left-0 w-full h-2 bg-gradient-to-r ${gradient} rounded-t-2xl`} />
                                <div className={`absolute bottom-0 right-0 w-2 h-full bg-gradient-to-b ${gradient} rounded-r-2xl`} />
                                <Icon className={`mx-auto mb-4 bg-gradient-to-r ${gradient} bg-clip-text text-transparent`} size={64} />
                                <h3 className="text-2xl font-bold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-4">{title}</h3>
                                <p className="text-green-900/90">{description}</p>
                            </div>
                        </div>
                    ))}
                </section>

                {/* About Section */}
                <section id="about" className="mb-20 pt-16 relative">
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(34,197,94,0.2),transparent_70%)]" />
                    <div className="relative backdrop-blur-xl bg-white/50 rounded-3xl shadow-2xl overflow-hidden border border-green-200/50">
                        <div className="absolute inset-0 bg-gradient-to-br from-green-100/95 via-emerald-100/95 to-teal-100/95" />
                        <div className="absolute top-0 left-0 right-0 h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600" />
                        <div className="relative p-12">
                            <h3 className="text-4xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent text-center mb-8">
                                About MetaCorp
                            </h3>
                            <div className="grid md:grid-cols-2 gap-12">
                                <div className="space-y-8">
                                    {[
                                        {
                                            title: "Our Mission",
                                            content: "MetaCorp is dedicated to revolutionizing business decision-making through advanced AI technology. We empower organizations to make data-driven decisions by simulating multiple future scenarios and identifying optimal paths forward."
                                        },
                                        {
                                            title: "Our Technology",
                                            content: "Powered by cutting-edge AI and machine learning algorithms, our platform processes vast amounts of data to create accurate, actionable insights for your business. We combine predictive analytics with real-time monitoring to provide comprehensive decision support."
                                        }
                                    ].map(({ title, content }, index) => (
                                        <div key={index} className="relative group">
                                            <div className="absolute inset-0 bg-gradient-to-br from-green-300/30 to-teal-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                                            <div className="relative bg-white/60 backdrop-blur-md p-6 rounded-2xl shadow-xl border border-green-200/50">
                                                <div className="absolute -inset-px bg-gradient-to-r from-green-500/30 via-emerald-500/30 to-teal-500/30 rounded-2xl blur" />
                                                <h4 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-4">{title}</h4>
                                                <p className="text-green-900 relative z-10">{content}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                <div className="grid grid-cols-2 gap-6">
                                    {[
                                        { Icon: Clock, title: "Real-time Analysis", description: "Instant insights for quick decisions" },
                                        { Icon: Target, title: "Precision Targeting", description: "Accurate predictions and insights" },
                                        { Icon: Lightbulb, title: "Innovation Focus", description: "Cutting-edge AI technology" },
                                        { Icon: BarChart2, title: "Data Visualization", description: "Clear, actionable insights" }
                                    ].map(({ Icon, title, description }, index) => (
                                        <div key={index} className="group relative">
                                            <div className="absolute inset-0 bg-gradient-to-br from-green-300/30 to-teal-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                                            <div className="relative bg-white/60 backdrop-blur-md p-6 rounded-2xl shadow-xl border border-green-200/50">
                                                <Icon className="mx-auto mb-2 text-transparent bg-gradient-to-r from-green-700 via-emerald-700 to-teal-700 bg-clip-text" size={40} />
                                                <h5 className="font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-2">{title}</h5>
                                                <p className="text-green-900 text-sm">{description}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* CTA Section */}
                <section id="simulation" className="text-center mb-20 pt-16 relative">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(34,197,94,0.2),transparent_70%)]" />
                    <div className="relative">
                        <h3 className="text-4xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent mb-6">
                            Ready to See Your Business's Future?
                        </h3>
                        <p className="text-xl text-green-900 max-w-2xl mx-auto mb-8 backdrop-blur-md bg-white/50 p-6 rounded-2xl shadow-xl border border-green-200/50">
                            Start your journey with MetaCorp's AI-powered business simulator and unlock 
                            insights that will transform your decision-making process.
                        </p>
                        <button 
                            onClick={() => navigate('/business-form')}
                            className="relative group px-12 py-4 rounded-full overflow-hidden"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-green-700 via-emerald-700 to-teal-700 transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="absolute inset-0 bg-[linear-gradient(45deg,rgba(255,255,255,0.15)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.15)_50%,rgba(255,255,255,0.15)_75%,transparent_75%,transparent)] bg-[length:24px_24px]" />
                            <span className="relative text-lg font-semibold text-white">Start Your Simulation</span>
                        </button>
                    </div>
                </section>
            </main>
        </div>
    );
};

export default Home;