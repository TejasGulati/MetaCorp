import React from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { ModeContext } from '../context/Mode';

const ParallelForm = () => {
    const {changeMode } = React.useContext(ModeContext);
    const navigate =useNavigate()
    const { register, control, handleSubmit } = useForm({
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
        changeMode("parallel");
        
        const processedData = {
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
            base_decisions: {
                hiring_rate: Number(data.base_decisions.hiring_rate),
                rd_investment: Number(data.base_decisions.rd_investment),
                market_expansion: Number(data.base_decisions.market_expansion)
            },
            decision_variations: data.decision_variations.map((variation) => ({
                hiring_rate: Number(variation.hiring_rate),
                rd_investment: Number(variation.rd_investment),
                market_expansion: Number(variation.market_expansion)
            })),
            num_years: Number(data.num_years),
            monte_carlo_sims: Number(data.monte_carlo_sims)
        };
    
        try {
            const response = await fetch('http://localhost:8000/api/simulate/parallel/', {
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
            console.log('Parallel simulation results:', result);
            
            navigate("/simulation-results");
        } catch (error) {
            console.error('Error running parallel simulation:', error);
        }
    
        console.log(processedData);
    };
    return (
        <div className="max-w-4xl mx-auto p-6 bg-gradient-to-br from-[#d1fae5] to-[#a7f3d0] rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-6 text-[#064e3b]">Parallel Simulation</h2>

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                <div className="grid grid-cols-2 gap-4 bg-white/50 p-6 rounded-lg">
                    {['name', 'industry', 'revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].map((field) => (
                        <div key={field}>
                            <label className="block text-[#065f46] font-medium mb-1">
                                {field.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                            </label>
                            <input
                                {...register(`company_data.${field}`)}
                                className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                                type={['revenues', 'profits', 'market_value', 'employees', 'revenue_growth', 'profit_margin', 'costs'].includes(field) ? 'number' : 'text'}
                            />
                        </div>
                    ))}
                </div>

                <h3 className=" text-xl font-semibold text-[#064e3b]">Base Decisions</h3>
                <div className="bg-white/50 p-6 grid grid-cols-3 gap-4bg-white/50 rounded-lg">
                    {['hiring_rate', 'rd_investment', 'market_expansion'].map((field) => (
                        <div key={field} className='bg-white/50 p-6 rounded-lg'>
                            <label className="block text-[#065f46] font-medium mb-1">
                                {field.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                            </label>
                            <input
                                {...register(`base_decisions.${field}`)}
                                className=" w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                                type="number"
                                step="0.01"
                            />
                        </div>
                    ))}
                </div>

                <h3 className=" text-xl font-semibold text-[#064e3b]">Decision Variations</h3>
                <div>
                    <button
                        type="button"
                        onClick={() => append({ hiring_rate: '', rd_investment: '', market_expansion: '' })}
                        className="bg-[#059669] text-white px-4 py-2 rounded hover:bg-[#065f46] mb-4"
                    >
                        Add Variation
                    </button>
                    {fields.map((field, index) => (
                        <div key={field.id} className="bg-white/50 rounded-lg mb-4 p-4 border border-[#059669]">
                            <div className="flex justify-between items-center">
                                <h4 className="text-[#064e3b] font-medium">Variation {index + 1}</h4>
                                <button
                                    type="button"
                                    onClick={() => remove(index)}
                                    className="text-red-500 hover:text-red-700"
                                >
                                    Remove
                                </button>
                            </div>
                            <div className=" grid grid-cols-3 gap-4">
                                {['hiring_rate', 'rd_investment', 'market_expansion'].map((variationField) => (
                                    <div key={variationField}>
                                        <label className="block text-[#065f46] font-medium mb-1">
                                            {variationField.replace('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                                        </label>
                                        <input
                                            {...register(`decision_variations.${index}.${variationField}`)}
                                            className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                                            type="number"
                                            step="0.01"
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="grid grid-cols-1 gap-4 bg-white/50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-[#064e3b]">Simulation Parameters</h3>
                    <div>
                        <label className="block text-[#065f46] font-medium mb-1">Number of Years</label>
                        <input
                            {...register('num_years')}
                            className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            type="number"
                        />
                    </div>
                    <div>
                        <label className="block text-[#065f46] font-medium mb-1">Monte Carlo Simulations</label>
                        <input
                            {...register('monte_carlo_sims')}
                            className="w-full p-2 border border-[#059669] rounded focus:outline-none focus:ring-2 focus:ring-[#059669]"
                            type="number"
                        />
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

export default ParallelForm;
