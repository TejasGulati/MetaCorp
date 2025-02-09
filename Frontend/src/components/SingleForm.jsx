import React from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { ModeContext } from '../context/Mode';
import { api } from '../utils/api';
import { ArrowRight, AlertCircle } from 'lucide-react';

const GradientBackground = () => (
    <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-emerald-50 to-teal-100" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(34,197,94,0.2),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.25),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(52,211,153,0.2),transparent_60%)]" />
        <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-white/40 to-transparent" />
    </div>
);

const SingleForm = () => {
    const { changeMode, setData } = React.useContext(ModeContext);
    const navigate = useNavigate();
    const { register, handleSubmit, formState: { errors } } = useForm({
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
        await changeMode("single");
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
            const result = await api.post('/simulate/', processedData);
            setData(result);
            console.log('Simulation results:', result);
            navigate("/simulation-results");
        } catch (error) {
            console.error('Error running simulation:', error);
        }
    };

    return (
        <div className="min-h-screen relative">
            <GradientBackground />
            
            <div className="relative max-w-6xl mx-auto p-8">
                <div className="relative backdrop-blur-xl bg-white/30 rounded-3xl shadow-2xl overflow-hidden border border-green-200/50 p-8">
                    <div className="absolute top-0 left-0 right-0 h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600" />
                    
                    <h2 className="text-4xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent mb-8">
                        Single Business Simulation
                    </h2>

                    <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
                        {/* Company Data Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-green-300/30 to-teal-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                    Company Information
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                    {['name', 'industry', 'revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].map((field) => (
                                        <div key={field} className="relative">
                                            <label className="block text-emerald-900 font-medium mb-2">
                                                {field.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                                            </label>
                                            <input
                                                {...register(`company_data.${field}`)}
                                                type={['revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].includes(field) ? 'number' : 'text'}
                                                step="0.01"
                                                className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Strategic Decisions Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-emerald-300/30 to-green-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                    Strategic Decisions
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                    {['hiring_rate', 'rd_investment', 'market_expansion'].map((field) => (
                                        <div key={field} className="relative">
                                            <label className="block text-emerald-900 font-medium mb-2">
                                                {field.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                                            </label>
                                            <input
                                                {...register(`decisions.${field}`)}
                                                type="number"
                                                step="0.01"
                                                className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Simulation Parameters Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-teal-300/30 to-emerald-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                    Simulation Parameters
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="relative">
                                        <label className="block text-emerald-900 font-medium mb-2">
                                            Number of Years
                                        </label>
                                        <input
                                            {...register("num_years")}
                                            type="number"
                                            className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                        />
                                    </div>
                                    <div className="relative">
                                        <label className="block text-emerald-900 font-medium mb-2">
                                            Market Scenario
                                        </label>
                                        <select
                                            {...register("market_scenario")}
                                            className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                        >
                                            <option value="baseline">Baseline</option>
                                            <option value="optimistic">Optimistic</option>
                                            <option value="pessimistic">Pessimistic</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            className="relative group w-full overflow-hidden"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="absolute inset-0 bg-[linear-gradient(45deg,rgba(255,255,255,0.15)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.15)_50%,rgba(255,255,255,0.15)_75%,transparent_75%,transparent)] bg-[length:24px_24px]" />
                            <div className="relative flex items-center justify-center space-x-2 py-4 text-lg font-semibold text-white">
                                <span>Submit Simulation</span>
                                <ArrowRight className="w-5 h-5" />
                            </div>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default SingleForm;