import React from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { ModeContext } from '../context/Mode';


const SingleForm = () => {
    const { changeMode,setData } = React.useContext(ModeContext);
    const navigate=useNavigate()
    const { register, handleSubmit } = useForm({
        defaultValues: {
            company_data: {
                name: '',
                industry: '',
                revenues: '',
                profits: '',
                market_value: '',
                employees: '',
                revenue_growth: '',
                profit_margin: '',
                costs: ''
            },
            decisions: {
                hiring_rate: '',
                rd_investment: '',
                market_expansion: ''
            },
            num_years: 5,
            market_scenario: 'baseline'
        }
    });

    const onSubmit = async (data) => {
        await changeMode("single")
        const processedData = {
            ...data,
            company_data: {
                ...data.company_data,
                revenues: Number(data.company_data.revenues),
                profits: Number(data.company_data.profits),
                market_value: Number(data.company_data.market_value),
                employees: Number(data.company_data.employees),
                revenue_growth: Number(data.company_data.revenue_growth),
                profit_margin: Number(data.company_data.profit_margin),
                costs: Number(data.company_data.costs)
            },
            decisions: {
                hiring_rate: Number(data.decisions.hiring_rate),
                rd_investment: Number(data.decisions.rd_investment),
                market_expansion: Number(data.decisions.market_expansion)
            },
            num_years: Number(data.num_years),
            market_scenario: data.market_scenario 
        };
        
        try {
            const response = await fetch('http://localhost:8000/api/simulate/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(processedData)
            });
    
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
    
            const result = await response.json();
            setData(result)
            navigate("/simulation-results");
        } catch (error) {
            console.error('Error running simulation:', error);
        }
    
        console.log(processedData);
    };

    return (
        <div className="max-w-4xl mx-auto p-6 bg-gradient-to-br from-[#d1fae5] to-[#a7f3d0] rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-6 text-[#064e3b]">Single Simulation Parameters</h2>
            
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">

                <div className="bg-white/50 p-6 rounded-lg">
                    <h3 className="text-xl font-semibold mb-4 text-[#064e3b]">Company Data</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Company Name</label>
                            <input
                                {...register("company_data.name")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Industry</label>
                            <input
                                {...register("company_data.industry")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Revenues</label>
                            <input
                                type="number"
                                {...register("company_data.revenues")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Profits</label>
                            <input
                                type="number"
                                {...register("company_data.profits")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Market Value</label>
                            <input
                                type="number"
                                {...register("company_data.market_value")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Employees</label>
                            <input
                                type="number"
                                {...register("company_data.employees")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Revenue Growth (%)</label>
                            <input
                                type="number"
                                step="0.1"
                                {...register("company_data.revenue_growth")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Profit Margin (%)</label>
                            <input
                                type="number"
                                step="0.1"
                                {...register("company_data.profit_margin")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Costs</label>
                            <input
                                type="number"
                                {...register("company_data.costs")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>
                    </div>
                </div>

                <div className="bg-white/50 p-6 rounded-lg">
                    <h3 className="text-xl font-semibold mb-4 text-[#064e3b]">Strategic Decisions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Hiring Rate</label>
                            <input
                                type="number"
                                step="0.01"
                                {...register("decisions.hiring_rate")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">R&D Investment</label>
                            <input
                                type="number"
                                step="0.01"
                                {...register("decisions.rd_investment")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Market Expansion</label>
                            <input
                                type="number"
                                step="0.01"
                                {...register("decisions.market_expansion")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>
                    </div>
                </div>
                <div className="bg-white/50 p-6 rounded-lg">
                    <h3 className="text-xl font-semibold mb-4 text-[#064e3b]">Simulation Parameters</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Number of Years</label>
                            <input
                                type="number"
                                {...register("num_years")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            />
                        </div>

                        <div>
                            <label className="block text-[#065f46] font-medium mb-1">Market Scenario</label>
                            <select
                                {...register("market_scenario")}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            >
                                <option value="baseline">Baseline</option>
                                <option value="optimistic">Optimistic</option>
                                <option value="pessimistic">Pessimistic</option>
                            </select>
                        </div>
                    </div>
                </div>
                <button
                    type="submit"
                    className="w-full bg-[#059669] hover:bg-[#065f46] text-white font-bold py-2 px-4 rounded transition-colors duration-200"
                >
                    Submit Simulation
                </button>
            </form>
        </div>
    );
};

export default SingleForm;