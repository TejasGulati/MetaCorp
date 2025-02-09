import React from 'react';

function Footer() {
    return (
        <footer className="relative overflow-hidden">
            {/* Background layers */}
            <div className="absolute inset-0 bg-gradient-to-r from-green-100 via-emerald-50 to-teal-100" />
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_150%,rgba(34,197,94,0.2),transparent_70%)]" />
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(16,185,129,0.15),transparent_60%)]" />
            <div className="absolute inset-0 bg-[linear-gradient(45deg,rgba(34,197,94,0.1)_25%,transparent_25%,transparent_50%,rgba(34,197,94,0.1)_50%,rgba(34,197,94,0.1)_75%,transparent_75%,transparent)] bg-[length:64px_64px] opacity-30" />
            
            {/* Top border gradient */}
            <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-green-200/50 via-emerald-300/50 to-teal-200/50" />

            {/* Content */}
            <div className="container mx-auto px-4 py-12 relative">
                <div className="text-center">
                    {/* Logo section with gradient text */}
                    <div className="relative inline-block group mb-6">
                        <h4 className="text-2xl font-bold bg-gradient-to-r from-green-800 via-emerald-700 to-teal-800 bg-clip-text text-transparent">
                            MetaCorp
                        </h4>
                        <div className="absolute -inset-x-6 -inset-y-4 bg-gradient-to-r from-green-500/10 via-emerald-500/10 to-teal-500/10 blur-lg group-hover:opacity-75 transition-opacity duration-500 opacity-0" />
                    </div>

                    {/* Tagline with backdrop blur card */}
                    <div className="relative max-w-md mx-auto mb-8">
                        <div className="absolute inset-0 bg-gradient-to-r from-green-300/20 via-emerald-300/20 to-teal-300/20 blur-xl" />
                        <p className="relative text-lg text-green-800 backdrop-blur-sm bg-white/30 py-3 px-6 rounded-full">
                            Powering Business Decisions with AI
                        </p>
                    </div>

                    {/* Links section */}
                    <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto mb-8">
                        {['Privacy Policy', 'Terms of Service', 'Contact Us'].map((text, index) => (
                            <button
                                key={index}
                                className="text-green-700 hover:text-green-900 transition-colors duration-300 relative group"
                            >
                                <span className="relative z-10">{text}</span>
                                <div className="absolute -inset-x-2 -inset-y-1 bg-gradient-to-r from-green-500/10 via-emerald-500/10 to-teal-500/10 rounded-lg blur-sm group-hover:opacity-100 transition-opacity duration-300 opacity-0" />
                            </button>
                        ))}
                    </div>

                    {/* Copyright with subtle gradient background */}
                    <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-green-200/20 via-emerald-200/20 to-teal-200/20 blur-md" />
                        <p className="relative text-sm text-green-700 font-medium">
                            © 2025 MetaCorp. All rights reserved.
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}

export default Footer;