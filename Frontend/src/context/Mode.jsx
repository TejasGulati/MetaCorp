import React, { createContext, useState } from 'react';


const ModeContext = createContext();


const ModeProvider = ({ children }) => {
    const [mode, setMode] = useState('parallel');
    
    const changeMode = (newMode) => {
        setMode(newMode);
    };
    
    return (
        <ModeContext.Provider value={{ mode, changeMode }}>
            {children}
        </ModeContext.Provider>
    );
};
export { ModeContext, ModeProvider };