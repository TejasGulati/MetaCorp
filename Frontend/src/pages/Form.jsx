import React from 'react';
import SingleForm from '../components/SingleForm';
import ParallelForm from '../components/ParallelForm';

const GradientBackground = () => (
    <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-emerald-50 to-teal-100" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(34,197,94,0.2),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.25),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(52,211,153,0.2),transparent_60%)]" />
    </div>
);

function Form() {
    const [activeForm, setActiveForm] = React.useState("single");

    return (
        <div className="min-h-screen relative overflow-hidden pt-20">
            <GradientBackground />
            
            <div className="container mx-auto px-4 relative">
                <div className="relative backdrop-blur-xl bg-white/50 rounded-3xl shadow-2xl overflow-hidden border border-green-200/50">
                    <div className="absolute inset-0 bg-gradient-to-br from-green-100/95 via-emerald-100/95 to-teal-100/95" />
                    <div className="absolute top-0 left-0 right-0 h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600" />
                    
                    <div className="relative p-8">
                        <div className="text-center mb-8">
                            <div className="inline-block">
                                <div className="relative">
                                    <h2 className="text-4xl font-extrabold bg-gradient-to-r from-green-900 via-emerald-700 to-teal-900 bg-clip-text text-transparent mb-6">
                                        Simulation Setup
                                    </h2>
                                    <div className="absolute -inset-1 bg-gradient-to-r from-green-500/30 via-emerald-500/30 to-teal-500/30 blur-3xl" />
                                </div>
                            </div>
                        </div>

                        <div className="flex justify-center items-center space-x-6 mb-8">
                            {['single', 'parallel'].map((type) => (
                                <button
                                    key={type}
                                    type="button"
                                    onClick={() => setActiveForm(activeForm === type ? null : type)}
                                    className="group relative"
                                >
                                    <div className={`absolute inset-0 bg-gradient-to-br ${
                                        activeForm === type
                                            ? 'from-green-600/80 via-emerald-600/80 to-teal-600/80'
                                            : 'from-green-400/30 via-emerald-400/30 to-teal-400/30'
                                    } rounded-xl blur-md transform group-hover:scale-110 transition-all duration-300`} />
                                    
                                    <div className={`relative px-6 py-3 rounded-xl backdrop-blur-md border border-green-200/50 transform transition-all duration-300 ${
                                        activeForm === type
                                            ? 'bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 text-white shadow-lg scale-105'
                                            : 'bg-white/80 text-green-800 hover:scale-105'
                                    }`}>
                                        <span className="font-semibold">
                                            {type.charAt(0).toUpperCase() + type.slice(1)} Simulation
                                        </span>
                                    </div>
                                </button>
                            ))}
                        </div>

                        <div className="relative">
                            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(34,197,94,0.1),transparent_70%)]" />
                            {activeForm === 'single' && (
                                <div className="relative transform transition-all duration-500 ease-in-out">
                                    <SingleForm />
                                </div>
                            )}
                            {activeForm === 'parallel' && (
                                <div className="relative transform transition-all duration-500 ease-in-out">
                                    <ParallelForm />
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Form;