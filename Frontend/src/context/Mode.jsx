import React, { createContext, useState } from 'react';


const ModeContext = createContext();


const ModeProvider = ({ children }) => {
    const [mode, setMode] = useState('parallel');
    const [data,setData]=useState({})
    
    const changeMode = (newMode) => {
        setMode(newMode);
    };
    
    return (
        <ModeContext.Provider value={{ mode, changeMode ,data,setData }}>
            {children}
        </ModeContext.Provider>
    );
};
export { ModeContext, ModeProvider };