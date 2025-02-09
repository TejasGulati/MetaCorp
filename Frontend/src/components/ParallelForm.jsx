import React from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { ModeContext } from '../context/Mode';
import { api } from '../utils/api';
import { Plus, Trash2, ArrowRight, AlertCircle,Loader2 } from 'lucide-react';

const GradientBackground = () => (
    <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-emerald-50 to-teal-100" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(34,197,94,0.2),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.25),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(52,211,153,0.2),transparent_60%)]" />
        <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-white/40 to-transparent" />
    </div>
);

const ParallelForm = () => {
    const [isLoading, setIsLoading] = React.useState(false);
    const { changeMode, setData } = React.useContext(ModeContext);
    const navigate = useNavigate();
    const { register, control, handleSubmit, formState: { errors } } = useForm({
        mode: 'onBlur',
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
            base_decisions: {
                hiring_rate: '',
                rd_investment: '',
                market_expansion: ''
            },
            decision_variations: [
                {
                    hiring_rate: '',
                    rd_investment: '',
                    market_expansion: ''
                }
            ],
            num_years: 5,
            monte_carlo_sims: 50
        }
    });

    const { fields, append, remove } = useFieldArray({
        control,
        name: 'decision_variations'
    });

    const onSubmit = async (data) => {
        setIsLoading(true)
        let processedData;
        try {
            processedData = {
                company_data: {
                    name: data.company_data.name,
                    industry: data.company_data.industry,
                    revenues: Number(data.company_data.revenues),
                    profits: Number(data.company_data.profits),
                    market_value: Number(data.company_data.market_value),
                    employees: Number(data.company_data.employees),
                    revenue_growth: Number(data.company_data.revenue_growth),
                    profit_margin: Number(data.company_data.profit_margin),
                    costs: Number(data.company_data.costs)
                },
                base_decisions: {
                    hiring_rate: Number(data.base_decisions.hiring_rate),
                    rd_investment: Number(data.base_decisions.rd_investment),
                    market_expansion: Number(data.base_decisions.market_expansion)
                },
                decision_variations: data.decision_variations.map(variation => ({
                    hiring_rate: Number(variation.hiring_rate),
                    rd_investment: Number(variation.rd_investment),
                    market_expansion: Number(variation.market_expansion)
                })),
                num_years: Number(data.num_years),
                monte_carlo_sims: Number(data.monte_carlo_sims)
            };

            await changeMode("parallel");
            const response = await api.post('/simulate/parallel/', processedData);
            setData(response);
            navigate("/simulation-results");
        } catch (error) {
            console.error('Error details:', error);
            alert(`Error running simulation: ${error.message}`);
        }
    };

    return (
        <div className="min-h-screen relative">
            <GradientBackground />
            
            <div className="relative max-w-6xl mx-auto p-8">
                <div className="relative backdrop-blur-xl bg-white/30 rounded-3xl shadow-2xl overflow-hidden border border-green-200/50 p-8">
                    <div className="absolute top-0 left-0 right-0 h-2 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600" />
                    
                    <h2 className="text-4xl font-bold bg-gradient-to-r from-green-900 via-emerald-800 to-teal-900 bg-clip-text text-transparent mb-8">
                        Parallel Business Simulation
                    </h2>

                    <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
                        {/* Company Data Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-green-300/30 to-teal-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                    Company Information
                                </h3>
                                <div className="grid grid-cols-3 gap-6">
                                    {['name', 'industry', 'revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].map((field) => (
                                        <div key={field} className="relative">
                                            <label className="block text-emerald-900 font-medium mb-2">
                                                {field.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                                            </label>
                                            <input
                                                {...register(`company_data.${field}`, {
                                                    required: 'This field is required',
                                                    valueAsNumber: ['revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].includes(field)
                                                })}
                                                className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                                type={['revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].includes(field) ? 'number' : 'text'}
                                                step="0.01"
                                            />
                                            {errors.company_data?.[field] && (
                                                <div className="absolute -bottom-6 left-0 flex items-center text-red-500 text-sm">
                                                    <AlertCircle className="w-4 h-4 mr-1" />
                                                    {errors.company_data[field].message}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Base Decisions Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-emerald-300/30 to-green-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                    Base Decisions
                                </h3>
                                <div className="grid grid-cols-3 gap-6">
                                    {['hiring_rate', 'rd_investment', 'market_expansion'].map((field) => (
                                        <div key={field} className="relative">
                                            <label className="block text-emerald-900 font-medium mb-2">
                                                {field.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                                            </label>
                                            <input
                                                {...register(`base_decisions.${field}`, {
                                                    required: 'This field is required',
                                                    valueAsNumber: true
                                                })}
                                                className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                                type="number"
                                                step="0.01"
                                            />
                                            {errors.base_decisions?.[field] && (
                                                <div className="absolute -bottom-6 left-0 flex items-center text-red-500 text-sm">
                                                    <AlertCircle className="w-4 h-4 mr-1" />
                                                    {errors.base_decisions[field].message}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Decision Variations Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-teal-300/30 to-emerald-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <div className="flex justify-between items-center mb-6">
                                    <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent">
                                        Decision Variations
                                    </h3>
                                    <button
                                        type="button"
                                        onClick={() => append({ hiring_rate: '', rd_investment: '', market_expansion: '' })}
                                        className="flex items-center px-4 py-2 bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-lg hover:from-green-700 hover:to-teal-700 transition-all duration-200"
                                    >
                                        <Plus className="w-4 h-4 mr-2" />
                                        Add Variation
                                    </button>
                                </div>
                                
                                <div className="space-y-6">
                                    {fields.map((field, index) => (
                                        <div key={field.id} className="relative group">
                                            <div className="absolute inset-0 bg-gradient-to-br from-green-200/30 to-teal-200/30 rounded-xl blur-lg transform group-hover:scale-105 transition-transform duration-300" />
                                            <div className="relative bg-white/70 backdrop-blur-md p-6 rounded-xl border border-green-200/50">
                                                <div className="flex justify-between items-center mb-4">
                                                    <h4 className="text-lg font-medium text-emerald-900">
                                                        Variation {index + 1}
                                                    </h4>
                                                    <button
                                                        type="button"
                                                        onClick={() => remove(index)}
                                                        className="flex items-center text-red-500 hover:text-red-700 transition-colors duration-200"
                                                    >
                                                        <Trash2 className="w-4 h-4 mr-1" />
                                                        Remove
                                                    </button>
                                                </div>
                                                <div className="grid grid-cols-3 gap-6">
                                                    {['hiring_rate', 'rd_investment', 'market_expansion'].map((variationField) => (
                                                        <div key={variationField} className="relative">
                                                            <label className="block text-emerald-900 font-medium mb-2">
                                                                {variationField.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                                                            </label>
                                                            <input
                                                                {...register(`decision_variations.${index}.${variationField}`, {
                                                                    required: 'This field is required',
                                                                    valueAsNumber: true
                                                                })}
                                                                className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                                                type="number"
                                                                step="0.01"
                                                            />
                                                            {errors.decision_variations?.[index]?.[variationField] && (
                                                                <div className="absolute -bottom-6 left-0 flex items-center text-red-500 text-sm">
                                                                    <AlertCircle className="w-4 h-4 mr-1" />
                                                                    {errors.decision_variations[index][variationField].message}
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Simulation Parameters Section */}
                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-br from-green-300/30 to-teal-300/30 rounded-2xl blur-xl transform group-hover:scale-105 transition-transform duration-300" />
                            <div className="relative bg-white/60 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-green-200/50">
                                <h3 className="text-2xl font-semibold bg-gradient-to-r from-green-900 to-teal-900 bg-clip-text text-transparent mb-6">
                                    Simulation Parameters
                                </h3>
                                <div className="grid grid-cols-2 gap-6">
                                    <div className="relative">
                                        <label className="block text-emerald-900 font-medium mb-2">
                                            Number of Years
                                        </label>
                                        <input
                                            {...register('num_years', {
                                                required: 'This field is required',
                                                valueAsNumber: true,
                                                min: { value: 1, message: 'Must be at least 1 year' }
                                            })}
                                            className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                            type="number"
                                        />
                                        {errors.num_years && (
                                            <div className="absolute -bottom-6 left-0 flex items-center text-red-500 text-sm">
                                                <AlertCircle className="w-4 h-4 mr-1" />
                                                {errors.num_years.message}
                                            </div>
                                        )}
                                    </div>
                                    <div className="relative">
                                        <label className="block text-emerald-900 font-medium mb-2">
                                            Monte Carlo Simulations
                                        </label>
                                        <input
                                            {...register('monte_carlo_sims', {
                                                required: 'This field is required',
                                                valueAsNumber: true,
                                                min: { value: 1, message: 'Must be at least 1 simulation' }
                                            })}
                                            className="w-full p-3 border border-emerald-200 rounded-lg bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-200"
                                            type="number"
                                        />
                                        {errors.monte_carlo_sims && (
                                            <div className="absolute -bottom-6 left-0 flex items-center text-red-500 text-sm">
                                                <AlertCircle className="w-4 h-4 mr-1" />
                                                {errors.monte_carlo_sims.message}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="relative group w-full overflow-hidden disabled:opacity-50"
        >
                    <div className="absolute inset-0 bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 transform group-hover:scale-105 transition-transform duration-300" />
                    <div className="absolute inset-0 bg-[linear-gradient(45deg,rgba(255,255,255,0.15)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.15)_50%,rgba(255,255,255,0.15)_75%,transparent_75%,transparent)] bg-[length:24px_24px]" />
                    <div className="relative flex items-center justify-center space-x-2 py-4 text-lg font-semibold text-white">
                {isLoading ? (
                <>
                <Loader2 className="animate-spin w-5 h-5" />
                <span>Submitting...</span>
                </>
            ) : (
                <>
                <span>Submit Simulation</span>
                <ArrowRight className="w-5 h-5" />
                </>
            )}
            </div>
            </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default ParallelForm;