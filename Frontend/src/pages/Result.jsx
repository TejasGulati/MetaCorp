import React, { useEffect, useState, useContext } from 'react';
import { ModeContext } from '../context/Mode';

const GradientBackground = () => (
    <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-emerald-50 to-teal-100" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(34,197,94,0.2),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.25),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(52,211,153,0.2),transparent_60%)]" />
    </div>
);

const Result = () => {
    const { mode, data } = useContext(ModeContext);
    const isParallel = (mode === "single") ? false : true;
    const simulationData = data;
    
    const getRealities = () => {
        const realities = [];
        Object.keys(simulationData?.results || {}).forEach((key) => {
            realities.push({ realityName: key, data: simulationData.results[key] });
        });
        return realities;
    };

    return (
        <div className="min-h-screen relative overflow-hidden mt-20">
            <GradientBackground />
            
            <main className="container mx-auto px-4 py-12 relative">
                <div className="text-center mb-12 relative">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(34,197,94,0.2),transparent_70%)]" />
                    <h1 className="text-5xl font-extrabold bg-gradient-to-r from-green-900 via-emerald-700 to-teal-900 bg-clip-text text-transparent mb-6 relative z-10">
                        Simulation Results
                    </h1>
                </div>

                <div className="relative backdrop-blur-xl bg-white/50 rounded-3xl shadow-2xl overflow-hidden border border-green-200/50 mb-8">
                    <div className="absolute inset-0 bg-gradient-to-br from-green-100/95 via-emerald-100/95 to-teal-100/95" />
                    <div className="absolute top-0 left-0 right-0 h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600" />
                    <div className="relative p-8">
                        <h2 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-4">Scenario</h2>
                        <p className="text-green-800">
                            <strong>Years Simulated:</strong> {simulationData?.input_data?.num_years || 'N/A'}
                        </p>
                    </div>
                </div>

                {isParallel ? (
                    <div id="parallel-simulation-results" className="space-y-8">
                        <h2 className="text-3xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent mb-6">
                            Parallel Simulation Results
                        </h2>

                        {getRealities().length > 0 ? (
                            getRealities().map((reality, index) => (
                                <div key={index} className="group relative">
                                    <div className="absolute inset-0 bg-gradient-to-br from-green-400/30 via-emerald-400/30 to-teal-400/30 rounded-2xl blur-2xl transform group-hover:scale-105 transition-transform duration-500" />
                                    <div className="relative bg-white/80 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                        <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 rounded-t-2xl" />
                                        <h3 className="text-2xl font-bold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                            {reality.realityName.replace('reality_', '').replace('_', ' ').toUpperCase()}
                                        </h3>

                                        <div className="overflow-x-auto">
                                            <table className="w-full">
                                                <thead>
                                                    <tr className="border-b-2 border-green-100">
                                                        <th className="p-4 text-left text-green-800 font-semibold">Year</th>
                                                        <th className="p-4 text-left text-green-800 font-semibold">Revenue</th>
                                                        <th className="p-4 text-left text-green-800 font-semibold">Profit Margin</th>
                                                        <th className="p-4 text-left text-green-800 font-semibold">Market Value</th>
                                                        <th className="p-4 text-left text-green-800 font-semibold">Employees</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {reality.data.map((data, yearIndex) => (
                                                        <tr key={yearIndex} className="border-b border-green-50">
                                                            <td className="p-4 text-green-800">{data.year}</td>
                                                            <td className="p-4 text-green-800">${data.revenues.toFixed(2)}</td>
                                                            <td className="p-4 text-green-800">{data.profit_margin.toFixed(2)}%</td>
                                                            <td className="p-4 text-green-800">${data.market_value.toFixed(2)}</td>
                                                            <td className="p-4 text-green-800">{data.employees.toFixed(0)}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <p className="text-green-800">No parallel simulation results available.</p>
                        )}
                    </div>
                ) : (
                    <div className="space-y-8">
                        <h2 className="text-3xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent mb-6">
                            Single Simulation Results
                        </h2>

                        <div className="group relative">
                            <div className="absolute inset-0 bg-gradient-to-br from-green-400/30 via-emerald-400/30 to-teal-400/30 rounded-2xl blur-2xl transform group-hover:scale-105 transition-transform duration-500" />
                            <div className="relative bg-white/80 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 rounded-t-2xl" />
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        <thead>
                                            <tr className="border-b-2 border-green-100">
                                                <th className="p-4 text-left text-green-800 font-semibold">Year</th>
                                                <th className="p-4 text-left text-green-800 font-semibold">Revenue</th>
                                                <th className="p-4 text-left text-green-800 font-semibold">Profit Margin</th>
                                                <th className="p-4 text-left text-green-800 font-semibold">Market Value</th>
                                                <th className="p-4 text-left text-green-800 font-semibold">Employees</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {simulationData?.results.map((result, index) => (
                                                <tr key={index} className="border-b border-green-50">
                                                    <td className="p-4 text-green-800">{result.year}</td>
                                                    <td className="p-4 text-green-800">${result.revenues.toFixed(2)}</td>
                                                    <td className="p-4 text-green-800">{result.profit_margin.toFixed(2)}%</td>
                                                    <td className="p-4 text-green-800">${result.market_value.toFixed(2)}</td>
                                                    <td className="p-4 text-green-800">{result.employees.toFixed(0)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        <div className="group relative">
                            <div className="absolute inset-0 bg-gradient-to-br from-green-400/30 via-emerald-400/30 to-teal-400/30 rounded-2xl blur-2xl transform group-hover:scale-105 transition-transform duration-500" />
                            <div className="relative bg-white/80 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 rounded-t-2xl" />
                                <h3 className="text-2xl font-bold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-4">Insights</h3>
                                <ul className="space-y-2">
                                    {simulationData?.insights.map((insight, index) => (
                                        <li key={index} className="text-green-800">{insight}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>
                )}

                <div className="mt-12 space-y-8">
                    <h2 className="text-3xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent mb-6">
                        Visualizations
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {['market_value_trajectories', 'revenue_growth_trajectories', 'profit_margin_trajectories', 'metric_correlations'].map((key, index) => (
                            <div key={index} className="group relative">
                                <div className="absolute inset-0 bg-gradient-to-br from-green-400/30 via-emerald-400/30 to-teal-400/30 rounded-2xl blur-2xl transform group-hover:scale-105 transition-transform duration-500" />
                                <div className="relative bg-white/80 backdrop-blur-md p-4 rounded-2xl shadow-xl border border-green-200/50">
                                    <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 rounded-t-2xl" />
                                    <img
                                        src={simulationData?.visualizations?.[key]}
                                        alt={key.split('_').join(' ')}
                                        className="rounded-lg w-full"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Result;