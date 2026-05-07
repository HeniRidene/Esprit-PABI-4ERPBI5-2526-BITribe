"use client";

import { Users, ShieldCheck, Bus, Leaf } from "lucide-react";

const LinkedinIcon = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect width="4" height="12" x="2" y="9" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);

const teamMembers = [
  { name: "Heni Ridene", linkedin: "https://www.linkedin.com/in/heni-ridene/" },
  { name: "Mohamed Sbissi", linkedin: "#" },
  { name: "Sirine Ben Chouikha", linkedin: "#" },
  { name: "Mohamed Amjed Chemchik", linkedin: "#" },
  { name: "Emna Baya Ben Romdhane", linkedin: "#" },
  { name: "Hammami Eya", linkedin: "#" }
];

export default function AboutUsPage() {
  return (
    <div className="h-full overflow-y-auto pr-2 pb-6 animate-fade-in-up">
      {/* Hero Section */}
      <div className="bg-white rounded-3xl p-8 md:p-12 mb-8 border border-outline-variant/20 shadow-premium relative overflow-hidden text-center">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-[100px] -z-10" />
        
        <div className="relative z-10 max-w-4xl mx-auto mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary/10 text-primary font-bold text-[12px] uppercase tracking-widest mb-6">
            <Users className="w-4 h-4" /> About Our Team
          </div>
          <h1 className="text-3xl md:text-5xl font-extrabold text-primary tracking-tight mb-6 leading-tight">
            Intelligent Decision-Making Suite for Urban Authorities
          </h1>
          <p className="text-[16px] md:text-lg text-outline-variant leading-relaxed max-w-3xl mx-auto">
            We transform complex mobility data into actionable insights. Designed specifically for transport authorities (e.g., Île-de-France Mobilités, RATP), our suite enhances network performance, attractiveness, and sustainability. Every visual is engineered to deliver critical insights in under 5 seconds.
          </p>
        </div>

        <div className="relative z-10 max-w-5xl mx-auto">
          <div className="rounded-3xl overflow-hidden border border-outline-variant/30 shadow-2xl bg-white p-2 md:p-3">
            <img src="/members.png" alt="Team Members" className="w-full h-auto rounded-2xl object-cover" />
          </div>
        </div>
      </div>

      {/* Strategic Roles */}
      <div className="mb-10">
        <h2 className="text-2xl font-bold text-primary mb-6">
           Empowering Key Decision-Makers
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-2xl p-6 border border-outline-variant/20 shadow-premium hover:-translate-y-1 transition-transform duration-300">
            <div className="w-12 h-12 rounded-xl bg-[#4355b9]/10 flex items-center justify-center mb-4">
              <Bus className="w-6 h-6 text-[#4355b9]" />
            </div>
            <h3 className="text-[18px] font-bold text-primary mb-2">Mobilities Director</h3>
            <p className="text-sm text-outline-variant leading-relaxed">
              Focuses on ensuring exceptional punctuality (Target &gt; 80%), optimizing commercial speed, and maximizing network capacity.
            </p>
          </div>
          
          <div className="bg-white rounded-2xl p-6 border border-outline-variant/20 shadow-premium hover:-translate-y-1 transition-transform duration-300">
            <div className="w-12 h-12 rounded-xl bg-secondary/10 flex items-center justify-center mb-4">
              <Leaf className="w-6 h-6 text-secondary" />
            </div>
            <h3 className="text-[18px] font-bold text-primary mb-2">Ecological Transition</h3>
            <p className="text-sm text-outline-variant leading-relaxed">
              Monitors carbon intensity (Target &lt; 0.10 kg/pass.km) and air quality to drive sustainability goals forward.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-6 border border-outline-variant/20 shadow-premium hover:-translate-y-1 transition-transform duration-300">
            <div className="w-12 h-12 rounded-xl bg-[#6b5c00]/10 flex items-center justify-center mb-4">
              <ShieldCheck className="w-6 h-6 text-[#6b5c00]" />
            </div>
            <h3 className="text-[18px] font-bold text-primary mb-2">Safety Manager</h3>
            <p className="text-sm text-outline-variant leading-relaxed">
              Maintains high safety standards by monitoring accident density (Target &lt; 10/km²) and reducing transit crime rates.
            </p>
          </div>
        </div>
      </div>

      {/* The Team */}
      <div className="bg-primary rounded-3xl p-8 text-white relative overflow-hidden shadow-glow-primary">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-white/5 rounded-full blur-[80px]" />
        
        <div className="relative z-10">
          <h2 className="text-2xl font-bold mb-2">
            Meet the Minds Behind the Platform
          </h2>
          <p className="text-sm text-primary-fixed-dim mb-8 max-w-2xl">
            Developed by a dedicated group of ERP-BI engineering students at ESPRIT (2025-2026), combining technical rigor with a passion for urban innovation.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {teamMembers.map((member, idx) => (
              <div key={idx} className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/20 flex items-center justify-between group hover:bg-white/20 transition-all">
                <span className="font-semibold text-white tracking-wide">{member.name}</span>
                <a 
                  href={member.linkedin} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-white/10 text-white text-[11px] font-bold hover:bg-[#0a66c2] hover:border-[#0a66c2] border border-white/20 transition-colors uppercase tracking-wider"
                >
                  <LinkedinIcon className="w-3.5 h-3.5" />
                  Contact Me
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
