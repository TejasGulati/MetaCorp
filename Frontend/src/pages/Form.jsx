import React from 'react'
import SingleForm from '../components/SingleForm'
import ParallelForm from '../components/ParallelForm'

function Form() {
  const [activeForm, setActiveForm] = React.useState("single"); 

  return (
      <div className="w-full mx-auto p-6 bg-gradient-to-br from-[#f0fdfa] to-[#bbf7d0] rounded-lg shadow-lg ">

          <div className='flex justify-center items-center'>

          <h2 className="text-2xl font-bold mb-6 text-[#064e3b]">Simulation Setup</h2>
          </div>
          <div className="flex justify-center items-center space-x-4 mb-6">
              <button
                  type="button"
                  onClick={() => setActiveForm(activeForm === 'single' ? null : 'single')}
                  className={`px-4 py-2 font-medium rounded-lg transition-colors duration-200 ${
                      activeForm === 'single'
                          ? 'bg-[#064e3b] text-white'
                          : 'bg-[#d1fae5] text-[#064e3b] hover:bg-[#a7f3d0]'
                  }`}
              >
                  Single Simulation
              </button>
              <button
                  type="button"
                  onClick={() => setActiveForm(activeForm === 'parallel' ? null : 'parallel')}
                  className={`px-4 py-2 font-medium rounded-lg transition-colors duration-200 ${
                      activeForm === 'parallel'
                          ? 'bg-[#064e3b] text-white'
                          : 'bg-[#d1fae5] text-[#064e3b] hover:bg-[#a7f3d0]'
                  }`}
              >
                  Parallel Simulation
              </button>
          </div>

          {activeForm === 'single' && <SingleForm />}
          {activeForm === 'parallel' && <ParallelForm />}
      </div>
  );
};


export default Form