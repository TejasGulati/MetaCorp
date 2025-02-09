import React, { useEffect, useState ,useContext} from 'react';
import { ModeContext } from '../context/Mode';

const Result = () => {
    const {mode}=useContext(ModeContext)
    const isParallel=(mode==="single")?false:true;
    const [simulationData,setSimulationData]=useState()

    console.log(isParallel,mode);
    useEffect(() => {
        const fetchResults = async () => {
            try {
                const endpoint = mode === 'parallel' 
                    ? 'http://localhost:8000/api/simulate/parallel/'
                    : 'http://localhost:8000/api/simulate/';
                
                const response = await fetch(endpoint, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                setSimulationData(data)
            } catch (error) {
                console.error('Error fetching simulation results:', error);;
            }
        };

        fetchResults();
    }, []);
    
    
    const getRealities = () => {
        const realities = [];
        console.log('Data Keys:', Object.keys(simulationData?.results || {}));
        

        Object.keys(simulationData?.results || {}).forEach((key) => {
            realities.push({ realityName: key, data: simulationData.results[key] });
        });
        return realities;
    };

    return (
        <div className="bg-green-100 min-h-screen font-sans p-8">
        <h1 className="text-3xl font-bold mb-6 text-center text-green-900">Simulation Results</h1>
    
        <div className="bg-white shadow-md p-6 rounded-lg mb-8">
            <h2 className="text-2xl font-semibold mb-4 text-green-800">Scenario</h2>
            <p className="text-green-800">
                <strong>Years Simulated:</strong> {simulationData?.input_data?.num_years || 'N/A'}
            </p>
        </div>
    
        {isParallel ? (
            <div id="parallel-simulation-results">
                <h2 className="text-xl font-bold mb-4 text-green-900">Parallel Simulation Results</h2>
    
                {getRealities().length > 0 ? (
                    getRealities().map((reality, index) => (
                        <div key={index} className="bg-white p-6 rounded-lg shadow-md mb-8">
                            <h3 className="text-lg font-bold mb-4 text-green-800">
                                {reality.realityName.replace('reality_', '').replace('_', ' ').toUpperCase()}
                            </h3>
    
                            <table className="w-full bg-white shadow-md rounded-lg mb-4">
                                <thead>
                                    <tr className="bg-green-100">
                                        <th className="p-4 text-left text-green-800">Year</th>
                                        <th className="p-4 text-left text-green-800">Revenue</th>
                                        <th className="p-4 text-left text-green-800">Profit Margin</th>
                                        <th className="p-4 text-left text-green-800">Market Value</th>
                                        <th className="p-4 text-left text-green-800">Employees</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {reality.data.map((data, yearIndex) => (
                                        <tr key={yearIndex}>
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
                    ))
                ) : (
                    <p>No parallel simulation results available.</p>
                )}
                <h2 className="text-xl font-bold mb-4 text-green-900">Visualizations</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <img
                        src={simulationData?.visualizations.market_value_trajectories}
                        alt="Market Value Trajectory"
                        className="rounded-lg shadow-md"
                    />
                    <img
                        src={simulationData?.visualizations.revenue_growth_trajectories}
                        alt="Revenue Growth Trajectory"
                        className="rounded-lg shadow-md"
                    />
                    <img
                        src={simulationData?.visualizations.profit_margin_trajectories}
                        alt="Profit Margin Trajectory"
                        className="rounded-lg shadow-md"
                    />
                    <img
                        src={simulationData?.visualizations.metric_correlations}
                        alt="Metric Correlations"
                        className="rounded-lg shadow-md"
                    />
                </div>
            </div>
        ) : (
            <div>
                <h2 className="text-xl font-bold mb-4 text-green-900">Single Simulation Results</h2>
                <table className="w-full bg-white shadow-md rounded-lg mb-8">
                    <thead>
                        <tr className="bg-green-100">
                            <th className="p-4 text-left text-green-800">Year</th>
                            <th className="p-4 text-left text-green-800">Revenue</th>
                            <th className="p-4 text-left text-green-800">Profit Margin</th>
                            <th className="p-4 text-left text-green-800">Market Value</th>
                            <th className="p-4 text-left text-green-800">Employees</th>
                        </tr>
                    </thead>
                    <tbody>
                        {simulationData?.results.map((result, index) => (
                            <tr key={index}>
                                <td className="p-4 text-green-800">{result.year}</td>
                                <td className="p-4 text-green-800">${result.revenues.toFixed(2)}</td>
                                <td className="p-4 text-green-800">{result.profit_margin.toFixed(2)}%</td>
                                <td className="p-4 text-green-800">${result.market_value.toFixed(2)}</td>
                                <td className="p-4 text-green-800">{result.employees.toFixed(0)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
    
                {/* Insights */}
                <div className="bg-green-50 p-4 rounded-lg text-green-700 font-semibold mb-8">
                    <h3 className="text-lg font-bold">Insights:</h3>
                    <ul className="list-disc pl-6">
                        {simulationData?.insights.map((insight, index) => (
                            <li key={index} className="text-green-800">{insight}</li>
                        ))}
                    </ul>
                </div>
    
                {/* Visualizations */}
                <h2 className="text-xl font-bold mb-4 text-green-900">Visualizations</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <img
                        src={simulationData?.visualizations.market_value_trajectories}
                        alt="Market Value Trajectory"
                        className="rounded-lg shadow-md"
                    />
                    <img
                        src={simulationData?.visualizations.revenue_growth_trajectories}
                        alt="Revenue Growth Trajectory"
                        className="rounded-lg shadow-md"
                    />
                    <img
                        src={simulationData?.visualizations.profit_margin_trajectories}
                        alt="Profit Margin Trajectory"
                        className="rounded-lg shadow-md"
                    />
                    <img
                        src={simulationData?.visualizations.metric_correlations}
                        alt="Metric Correlations"
                        className="rounded-lg shadow-md"
                    />
                </div>
            </div>
        )}
    </div>
    
    );
};

export default Result;