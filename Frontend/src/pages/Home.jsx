import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    Globe, 
    Shield, 
    Brain,
    Target,
    BarChart2,
    Users,
    Clock,
    LineChart,
    Database,
    Lightbulb
} from 'lucide-react';
import Navbar from '../components/Navbar';

const Home = () => {
    const navigate = useNavigate();

    const handleStartSimulation = () => {
        navigate('/business-form');
    };

    return (
        <div className="min-h-screen">
            <main className="container mx-auto px-4 py-12">
                <section id="home" className="text-center mb-20 pt-16">
                    <h2 className="text-5xl font-extrabold text-green-900 mb-6">
                        Shape Your Business Future with AI
                    </h2>
                    <p className="text-xl text-green-800 max-w-3xl mx-auto mb-8">
                        Experience the power of MetaCorp's AI-driven business simulator that creates
                        parallel business realities, helping you foresee risks, identify opportunities,
                        and optimize your growth strategies with unprecedented accuracy.
                    </p>
                </section>

                <section className="grid md:grid-cols-3 gap-8 mb-20">
                    <div className="bg-gradient-to-br from-green-100 to-emerald-200 p-8 rounded-xl shadow-lg text-center transform hover:scale-105 transition-transform">
                        <Globe className="mx-auto mb-4 text-green-600" size={64} />
                        <h3 className="text-2xl font-bold text-green-900 mb-4">Multiverse Simulation</h3>
                        <p className="text-green-800">
                            Generate multiple future scenarios simultaneously, showing how decisions ripple across time and departments.
                        </p>
                    </div>

                    <div className="bg-gradient-to-br from-green-100 to-emerald-200 p-8 rounded-xl shadow-lg text-center transform hover:scale-105 transition-transform">
                        <Brain className="mx-auto mb-4 text-green-600" size={64} />
                        <h3 className="text-2xl font-bold text-green-900 mb-4">AI-Powered Insights</h3>
                        <p className="text-green-800">
                            Leverage advanced AI to forecast trends, identify patterns, and uncover hidden opportunities in your data.
                        </p>
                    </div>

                    <div className="bg-gradient-to-br from-green-100 to-emerald-200 p-8 rounded-xl shadow-lg text-center transform hover:scale-105 transition-transform">
                        <Shield className="mx-auto mb-4 text-green-600" size={64} />
                        <h3 className="text-2xl font-bold text-green-900 mb-4">Risk Detection</h3>
                        <p className="text-green-800">
                            Real-time anomaly detection and risk assessment to protect your business from potential threats.
                        </p>
                    </div>

                    <div className="bg-gradient-to-br from-green-100 to-emerald-200 p-8 rounded-xl shadow-lg text-center transform hover:scale-105 transition-transform">
                        <LineChart className="mx-auto mb-4 text-green-600" size={64} />
                        <h3 className="text-2xl font-bold text-green-900 mb-4">Advanced Analytics</h3>
                        <p className="text-green-800">
                            Deep dive into your data with sophisticated analytical tools and predictive modeling.
                        </p>
                    </div>

                    <div className="bg-gradient-to-br from-green-100 to-emerald-200 p-8 rounded-xl shadow-lg text-center transform hover:scale-105 transition-transform">
                        <Database className="mx-auto mb-4 text-green-600" size={64} />
                        <h3 className="text-2xl font-bold text-green-900 mb-4">Data Integration</h3>
                        <p className="text-green-800">
                            Seamlessly integrate with your existing data sources for comprehensive analysis.
                        </p>
                    </div>

                    <div className="bg-gradient-to-br from-green-100 to-emerald-200 p-8 rounded-xl shadow-lg text-center transform hover:scale-105 transition-transform">
                        <Users className="mx-auto mb-4 text-green-600" size={64} />
                        <h3 className="text-2xl font-bold text-green-900 mb-4">Team Collaboration</h3>
                        <p className="text-green-800">
                            Enable team-wide decision making with shared insights and collaborative analysis.
                        </p>
                    </div>
                </section>

                <section id="about" className="mb-20 pt-16 bg-gradient-to-br from-blue-50 to-green-50">
                    <div className="rounded-xl shadow-lg p-12">
                        <h3 className="text-4xl font-bold text-green-900 text-center mb-8">
                            About MetaCorp
                        </h3>
                        <div className="grid md:grid-cols-2 gap-12">
                            <div>
                                <h4 className="text-2xl font-semibold text-green-800 mb-4">Our Mission</h4>
                                <p className="text-green-700 mb-6">
                                    MetaCorp is dedicated to revolutionizing business decision-making through 
                                    advanced AI technology. We empower organizations to make data-driven 
                                    decisions by simulating multiple future scenarios and identifying optimal 
                                    paths forward.
                                </p>
                                <h4 className="text-2xl font-semibold text-green-800 mb-4">Our Technology</h4>
                                <p className="text-green-700">
                                    Powered by cutting-edge AI and machine learning algorithms, our platform 
                                    processes vast amounts of data to create accurate, actionable insights 
                                    for your business. We combine predictive analytics with real-time 
                                    monitoring to provide comprehensive decision support.
                                </p>
                            </div>
                            <div className="grid grid-cols-2 gap-6">
                                <div className="text-center">
                                    <Clock className="mx-auto mb-2 text-green-600" size={40} />
                                    <h5 className="font-semibold text-green-800 mb-2">Real-time Analysis</h5>
                                    <p className="text-green-700 text-sm">Instant insights for quick decisions</p>
                                </div>
                                <div className="text-center">
                                    <Target className="mx-auto mb-2 text-green-600" size={40} />
                                    <h5 className="font-semibold text-green-800 mb-2">Precision Targeting</h5>
                                    <p className="text-green-700 text-sm">Accurate predictions and insights</p>
                                </div>
                                <div className="text-center">
                                    <Lightbulb className="mx-auto mb-2 text-green-600" size={40} />
                                    <h5 className="font-semibold text-green-800 mb-2">Innovation Focus</h5>
                                    <p className="text-green-700 text-sm">Cutting-edge AI technology</p>
                                </div>
                                <div className="text-center">
                                    <BarChart2 className="mx-auto mb-2 text-green-600" size={40} />
                                    <h5 className="font-semibold text-green-800 mb-2">Data Visualization</h5>
                                    <p className="text-green-700 text-sm">Clear, actionable insights</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <section id="simulation" className="text-center mb-20 pt-16">
                    <h3 className="text-3xl font-bold text-green-900 mb-6">
                        Ready to See Your Business's Future?
                    </h3>
                    <p className="text-xl text-green-800 max-w-2xl mx-auto mb-8">
                        Start your journey with MetaCorp's AI-powered business simulator and unlock 
                        insights that will transform your decision-making process.
                    </p>
                    <button 
                        onClick={handleStartSimulation}
                        className="bg-green-600 text-white px-12 py-4 rounded-full hover:bg-green-700 
                            transition-all transform hover:scale-105 font-semibold text-lg shadow-lg"
                    >
                        Start Your Simulation
                    </button>
                </section>
            </main>
        </div>
    );
};

export default Home;