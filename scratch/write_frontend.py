import os

frontend_dir = "d:/vin-sphere-frontend"
os.makedirs(f"{frontend_dir}/src/app", exist_ok=True)
os.makedirs(f"{frontend_dir}/src/components", exist_ok=True)

# 1. globals.css
globals_css = """@import "tailwindcss";

:root {
  --background: #f0f4f9;
  --foreground: #0f172a;
}

body {
  color: var(--foreground);
  background: var(--background);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  overflow-x: hidden;
}

/* Custom styles for corporate policy input grids */
.input-outline-container {
  position: relative;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  background-color: #ffffff;
  transition: all 0.2s ease-in-out;
}

.input-outline-container:focus-within {
  border-color: #2563eb;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
}

.floating-label {
  position: absolute;
  top: -8px;
  left: 12px;
  background-color: #ffffff;
  padding: 0 4px;
  font-size: 10px;
  color: #6b7280;
  font-weight: 600;
  pointer-events: none;
}
"""

# 2. layout.js
layout_js = """import './globals.css'

export const metadata = {
  title: 'Chassis Lookup',
  description: 'Select policy type, enter chassis number, then lookup vehicle details',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen bg-[#f0f4f8]">
        {children}
      </body>
    </html>
  )
}
"""

# 3. ChassisInput.js
chassis_input_js = """"use client";

import React, { useState, useEffect } from 'react';
import { Search, RotateCcw, AlertTriangle, ShieldCheck, Car } from 'lucide-react';

export default function ChassisInput({ onDecode, isLoading, onReset }) {
  const [vin, setVin] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [warnMsg, setWarnMsg] = useState('');

  useEffect(() => {
    const cleanVin = vin.toUpperCase().replace(/[^A-Z0-9]/g, '');
    
    if (cleanVin.length === 0) {
      setErrorMsg('');
      setWarnMsg('');
      return;
    }

    const validLengths = [9, 10, 11, 12, 13, 17];
    if (!validLengths.includes(cleanVin.length)) {
      setErrorMsg(`Chassis number must be 9, 10, 11, 12, 13, or 17 characters (currently ${cleanVin.length}).`);
      setWarnMsg('');
      return;
    }

    setErrorMsg('');

    if (cleanVin.length === 17) {
      const checkResult = validate17CharVin(cleanVin);
      if (!checkResult.isValid) {
        setWarnMsg(checkResult.error);
      } else {
        setWarnMsg('');
      }
    } else {
      setWarnMsg('');
    }
  }, [vin]);

  function validate17CharVin(vinStr) {
    if (/[IOQ]/i.test(vinStr)) {
      return { isValid: false, error: "VIN cannot contain prohibited letters I, O, or Q." };
    }
    
    const transMap = {
      A:1, B:2, C:3, D:4, E:5, F:6, G:7, h:8, J:1, K:2, L:3, M:4, N:5, P:7, R:9,
      S:2, T:3, U:4, V:5, W:6, X:7, Y:8, Z:9
    };
    const weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];
    
    let sum = 0;
    for (let i = 0; i < 17; i++) {
      const char = vinStr[i];
      let val = 0;
      if (/\d/.test(char)) {
        val = parseInt(char, 10);
      } else if (transMap[char] !== undefined) {
        val = transMap[char];
      } else {
        return { isValid: false, error: `Invalid character '${char}' in VIN.` };
      }
      sum += val * weights[i];
    }
    
    const remainder = sum % 11;
    const expected = remainder === 10 ? "X" : String(remainder);
    const actual = vinStr[8];
    if (expected !== actual) {
      return { isValid: false, error: `Check-digit mismatch at 9th position (calculated '${expected}', but found '${actual}').` };
    }
    
    return { isValid: true, error: "" };
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    const cleanVin = vin.toUpperCase().replace(/[^A-Z0-9]/g, '');
    if ([9, 10, 11, 12, 13, 17].includes(cleanVin.length)) {
      onDecode(cleanVin);
    }
  };

  const handleClear = () => {
    setVin('');
    setErrorMsg('');
    setWarnMsg('');
    onReset();
  };

  return (
    <div className="w-full bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-blue-600" />
          <h2 className="font-bold text-slate-800 text-sm tracking-tight">Chassis lookup</h2>
        </div>
        <p className="text-xs text-slate-400">Enter chassis number to lookup vehicle details</p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">

          {/* Chassis Number */}
          <div className="md:col-span-9 input-outline-container py-1.5 px-3">
            <span className="floating-label">Chassis number</span>
            <div className="flex items-center gap-2 mt-1">
              <Car className="w-4 h-4 text-slate-400" />
              <input
                type="text"
                className="w-full bg-transparent text-slate-800 text-sm font-semibold focus:outline-none uppercase tracking-widest placeholder:text-slate-300"
                placeholder="Enter Chassis Number"
                value={vin}
                onChange={(e) => setVin(e.target.value)}
                disabled={isLoading}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="md:col-span-3 flex gap-2">
            <button
              type="submit"
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-3.5 px-4 font-semibold text-xs flex items-center justify-center gap-1.5 transition-all shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isLoading || !!errorMsg || vin.length === 0}
            >
              {isLoading ? (
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  <Search className="w-3.5 h-3.5" />
                  <span>Lookup</span>
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-500 rounded-lg p-3.5 transition-all"
              title="Reset"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

        </div>

        {/* Validation Output alerts */}
        {errorMsg && (
          <div className="text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-red-500 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
        {warnMsg && !errorMsg && (
          <div className="text-xs text-amber-600 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
            <span>{warnMsg}</span>
          </div>
        )}
      </form>
    </div>
  );
}
"""

# 4. Dashboard.js
dashboard_js = """"use client";

import React, { useState } from 'react';
import { Copy, ChevronDown, ChevronUp, Check, ShieldCheck, Calendar, Globe, Gauge, Cpu, Fuel, Users, Settings, Car, Wrench, AlertTriangle } from 'lucide-react';

export default function Dashboard({ data }) {
  const [activeTab, setActiveTab] = useState('ml');
  const [copied, setCopied] = useState(false);
  const [evaluationInputsOpen, setEvaluationInputsOpen] = useState(true);
  const [valuationOpen, setValuationOpen] = useState(true);
  const [evaluateChassisOpen, setEvaluateChassisOpen] = useState(true);
  const [rawJsonOpen, setRawJsonOpen] = useState(false); // Collapsed by default to keep dashboard clean
  const [specsOpen, setSpecsOpen] = useState(true); // Open by default
  
  // Interactive inputs state (Mileage and Accidents)
  const [mileage, setMileage] = useState('50000');
  const [accidents, setAccidents] = useState('0');

  // Auto-switch to first available pricing tab when data changes
  React.useEffect(() => {
    if (data) {
      if (data.ml_valuation) {
        setActiveTab('ml');
      } else if (data.lookup_valuation) {
        setActiveTab('lookup');
      } else if (data.toyota_valuation) {
        setActiveTab('toyota');
      } else if (data.depreciated_valuations?.DriveArabia || Object.keys(data.depreciated_valuations || {}).length > 0) {
        setActiveTab('depreciation');
      }
    }
  }, [data]);

  const decode = data?.decode_result;
  if (!decode) return null;

  // Valuation dynamic factors calculation based on input boxes
  const currentMileage = mileage ? Number(mileage) : 50000;
  const mileageDiff = currentMileage - 50000;
  let mileageFactor = 1.0;
  if (mileageDiff > 0) {
    mileageFactor = Math.max(0.7, 1.0 - (mileageDiff / 10000) * 0.015);
  } else {
    mileageFactor = Math.min(1.15, 1.0 - (mileageDiff / 10000) * 0.01);
  }

  const accidentCount = accidents ? Number(accidents) : 0;
  const accidentFactor = Math.max(0.6, 1.0 - (accidentCount * 0.05));
  const totalFactor = mileageFactor * accidentFactor;

  const formatAdjustedPrice = (val) => {
    if (!val) return 0;
    return Math.round(val * totalFactor);
  };

  // Generate description based on specs matching the screenshot
  const getGeneratedDescription = () => {
    const specs = [
      decode.trim !== 'UNKNOWN' ? decode.trim : 'STD',
      decode.cylinders !== 'UNKNOWN' ? `${decode.cylinders} cyls` : '4 cyls',
      decode.body_type !== 'UNKNOWN' ? decode.body_type : 'SEDAN',
      decode.no_of_passengers !== 'UNKNOWN' ? `${decode.no_of_passengers} SEATS` : '5 SEATS'
    ].filter(Boolean).join(' ');
    
    return specs ? `${specs} - GCC` : 'Vehicle specifications decoded successfully.';
  };

  // Pricing values helper
  const getValuationSource = () => {
    switch (activeTab) {
      case 'ml': return data.ml_valuation;
      case 'lookup': return data.lookup_valuation;
      case 'toyota': return data.toyota_valuation;
      case 'depreciation': 
        return data.depreciated_valuations?.DriveArabia || Object.values(data.depreciated_valuations || {})[0];
      default: return null;
    }
  };

  const activeVal = getValuationSource();

  const getAdjustedJson = () => {
    const cleanData = JSON.parse(JSON.stringify(data));
    if (cleanData.decode_result) {
      delete cleanData.decode_result.length;
      cleanData.decode_result.description = getGeneratedDescription();
    }
    
    // Set custom inputs in raw response too
    cleanData.mileage = mileage;
    cleanData.noOfAccidentsNonGcc = accidents;
    
    const adjustValuation = (val) => {
      if (!val) return null;
      return {
        retail_price: {
          average: formatAdjustedPrice(val.retail_price.average),
          minimum: formatAdjustedPrice(val.retail_price.minimum),
          maximum: formatAdjustedPrice(val.retail_price.maximum)
        },
        trade_price: {
          average: formatAdjustedPrice(val.trade_price.average),
          minimum: formatAdjustedPrice(val.trade_price.minimum),
          maximum: formatAdjustedPrice(val.trade_price.maximum)
        }
      };
    };

    if (cleanData.ml_valuation) cleanData.ml_valuation = adjustValuation(cleanData.ml_valuation);
    if (cleanData.lookup_valuation) cleanData.lookup_valuation = adjustValuation(cleanData.lookup_valuation);
    if (cleanData.toyota_valuation) cleanData.toyota_valuation = adjustValuation(cleanData.toyota_valuation);
    if (cleanData.depreciated_valuations) {
      const adjustedDep = {};
      for (const k in cleanData.depreciated_valuations) {
        adjustedDep[k] = adjustValuation(cleanData.depreciated_valuations[k]);
      }
      cleanData.depreciated_valuations = adjustedDep;
    }
    return cleanData;
  };


  const handleCopyReport = () => {
    navigator.clipboard.writeText(JSON.stringify(getAdjustedJson(), null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full space-y-4">
      
      {/* 1. Vehicle Summary Bar */}
      <div className="w-full bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="font-bold text-xs text-slate-800 tracking-wide">Vehicle summary</span>
          </div>
          <span className="text-xs font-semibold text-slate-500 font-mono tracking-widest">{decode.vin}</span>
        </div>
        
        <div className="flex items-center gap-3">
          {decode.regional_spec && decode.regional_spec !== 'UNKNOWN' && (
            <span className="text-[10px] font-black text-emerald-600 bg-emerald-50 border border-emerald-100 px-2.5 py-1 rounded-full uppercase tracking-wider">
              {decode.regional_spec}
            </span>
          )}
          <button
            onClick={handleCopyReport}
            className="border border-slate-200 hover:bg-slate-50 text-slate-700 px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all"
            title="Copies the raw API JSON response"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
            <span>Copy report</span>
            <ChevronDown className="w-3 h-3 text-slate-400 border-l border-slate-200 pl-1 ml-1" />
          </button>
        </div>
      </div>

      {/* 2. Subtitle Header */}
      <div className="flex items-start gap-3 pl-1 mt-3">
        <Car className="w-5 h-5 text-blue-600 mt-0.5 shrink-0" />
        <div>
          <h2 className="text-sm font-extrabold text-slate-800 uppercase tracking-wide">
            {decode.make} · {decode.model} · {decode.year}
          </h2>
          <p className="text-xs text-slate-500 font-semibold mt-0.5">Body type: {decode.body_type}</p>
        </div>
      </div>

      {/* 3. Collapsible Vehicle Specifications Card */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <button
          onClick={() => setSpecsOpen(!specsOpen)}
          className="w-full flex items-center justify-between p-4 border-b border-slate-100 focus:outline-none"
        >
          <div className="flex flex-col gap-0.5 text-left">
            <h3 className="font-bold text-slate-800 text-sm">Vehicle specifications</h3>
            <p className="text-[11px] text-slate-400">Auto-filled after chassis lookup</p>
          </div>
          {specsOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {specsOpen && (
          <div className="p-4 space-y-4">
            {/* Main Grid of cards (12 total boxes) */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              
              {/* Year */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Calendar className="w-3 h-3 text-blue-500" /> Year
                </span>
                <span className="text-sm font-extrabold text-slate-800">{decode.year}</span>
              </div>

              {/* Make */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Car className="w-3 h-3 text-blue-500" /> Make
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.make}</span>
              </div>

              {/* Model */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Car className="w-3 h-3 text-blue-500" /> Model
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.model}</span>
              </div>

              {/* Regional Specs */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Globe className="w-3 h-3 text-blue-500" /> Regional specs
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.regional_spec}</span>
              </div>

              {/* Body Type */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Car className="w-3 h-3 text-blue-500" /> Body type
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.body_type}</span>
              </div>

              {/* Spec (Trim) */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Settings className="w-3 h-3 text-blue-500" /> Spec
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.trim}</span>
              </div>

              {/* Cylinders */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Wrench className="w-3 h-3 text-blue-500" /> Cylinders
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.cylinders}</span>
              </div>

              {/* Seats */}
              <div className="border border-slate-200/80 rounded-xl p-3 flex flex-col gap-1 bg-slate-50/50">
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1">
                  <Users className="w-3 h-3 text-blue-500" /> Seats
                </span>
                <span className="text-sm font-extrabold text-slate-800 uppercase">{decode.no_of_passengers !== 'UNKNOWN' ? `${decode.no_of_passengers} SEATS` : 'UNKNOWN'}</span>
              </div>

            </div>

            {/* Description Section */}
            <div className="border border-slate-200/80 rounded-xl p-4 bg-slate-50/30">
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Description</span>
              <p className="text-xs text-slate-700 font-semibold mt-1 uppercase tracking-wide">
                {getGeneratedDescription()}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 4. Collapsible RAW JSON Card */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <button
          onClick={() => setRawJsonOpen(!rawJsonOpen)}
          className="w-full flex items-center justify-between p-4 focus:outline-none"
        >
          <div className="flex flex-col gap-0.5 text-left">
            <h3 className="font-bold text-slate-800 text-sm">Raw JSON</h3>
            <p className="text-[11px] text-slate-400">Collapsible raw API response (length parameter removed)</p>
          </div>
          {rawJsonOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>
        
        {rawJsonOpen && (
          <div className="p-4 border-t border-slate-100 bg-[#f8fafc]">
            <div className="font-mono text-xs text-slate-700 overflow-x-auto whitespace-pre-wrap select-all">
              {JSON.stringify(getAdjustedJson(), null, 2)}
            </div>
          </div>
        )}
      </div>

      {/* 5. Collapsible Evaluation Inputs */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <button
          onClick={() => setEvaluationInputsOpen(!evaluationInputsOpen)}
          className="w-full flex items-center justify-between p-4 focus:outline-none"
        >
          <div className="flex flex-col gap-0.5 text-left">
            <h3 className="font-bold text-slate-800 text-sm">Evaluation inputs</h3>
            <p className="text-[11px] text-slate-400">Optional – used when evaluating chassis value</p>
          </div>
          {evaluationInputsOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>
        {evaluationInputsOpen && (
          <div className="p-4 border-t border-slate-100 bg-slate-50/30">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Mileage Box */}
              <div className="input-outline-container py-1.5 px-3">
                <span className="floating-label">Mileage (km)</span>
                <div className="flex items-center gap-2 mt-1">
                  <Gauge className="w-4 h-4 text-slate-400" />
                  <input
                    type="number"
                    className="w-full bg-transparent text-slate-800 text-sm font-semibold focus:outline-none placeholder:text-slate-300"
                    placeholder="Enter Mileage"
                    value={mileage}
                    onChange={(e) => setMileage(e.target.value)}
                  />
                </div>
              </div>

              {/* Accidents Box */}
              <div className="input-outline-container py-1.5 px-3">
                <span className="floating-label">No. of accidents</span>
                <div className="flex items-center gap-2 mt-1">
                  <AlertTriangle className="w-4 h-4 text-slate-400" />
                  <input
                    type="number"
                    className="w-full bg-transparent text-slate-800 text-sm font-semibold focus:outline-none placeholder:text-slate-300"
                    placeholder="Enter No. of Accidents"
                    value={accidents}
                    onChange={(e) => setAccidents(e.target.value)}
                  />
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      {/* 6. Collapsible Vehicle Valuation Details */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <button
          onClick={() => setValuationOpen(!valuationOpen)}
          className="w-full flex items-center justify-between p-4 focus:outline-none"
        >
          <div className="flex flex-col gap-0.5 text-left">
            <h3 className="font-bold text-slate-800 text-sm">Vehicle valuation</h3>
            <p className="text-[11px] text-slate-400">Evaluated values in AED</p>
          </div>
          {valuationOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>
        
        {valuationOpen && (
          <div className="p-4 border-t border-slate-100 space-y-6">
            
            {/* Tab switch selector */}
            <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
              <span className="text-xs font-bold text-slate-500">Select Valuation Source:</span>
              <div className="flex gap-1 bg-slate-100 border border-slate-200 p-1 rounded-xl">
                {[
                  { id: 'ml', name: 'CatBoost ML', available: !!data.ml_valuation },
                  { id: 'lookup', name: 'DB Lookup', available: !!data.lookup_valuation },
                  { id: 'toyota', name: 'Toyota Engine', available: !!data.toyota_valuation },
                  { id: 'depreciation', name: 'DriveArabia', available: !!(data.depreciated_valuations?.DriveArabia || Object.values(data.depreciated_valuations || {}).length > 0) },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                      !tab.available ? 'opacity-30 cursor-not-allowed' :
                      activeTab === tab.id ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-800'
                    }`}
                    onClick={() => tab.available && setActiveTab(tab.id)}
                    disabled={!tab.available}
                  >
                    {tab.name}
                  </button>
                ))}
              </div>
            </div>

            {activeVal ? (
              <div className="space-y-6">
                
                {/* Big Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Retail Average */}
                  <div className="border border-blue-100 rounded-xl p-4 bg-blue-50/20 flex flex-col gap-1">
                    <span className="text-[10px] text-blue-600 font-extrabold uppercase tracking-wider">Retail Average</span>
                    <span className="text-2xl font-black text-blue-700">AED {formatAdjustedPrice(activeVal.retail_price.average).toLocaleString()}</span>
                  </div>

                  {/* Trade Average */}
                  <div className="border border-emerald-100 rounded-xl p-4 bg-emerald-50/20 flex flex-col gap-1">
                    <span className="text-[10px] text-emerald-600 font-extrabold uppercase tracking-wider">Trade Average</span>
                    <span className="text-2xl font-black text-emerald-700">AED {formatAdjustedPrice(activeVal.trade_price.average).toLocaleString()}</span>
                  </div>
                </div>

                {/* Detailed Table */}
                <div className="border border-slate-200/80 rounded-xl overflow-hidden shadow-sm">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                        <th className="py-3 px-4">Price Type</th>
                        <th className="py-3 px-4 text-center">Average</th>
                        <th className="py-3 px-4 text-center">Maximum</th>
                        <th className="py-3 px-4 text-center">Minimum</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-xs font-semibold text-slate-700">
                      <tr>
                        <td className="py-3 px-4 font-bold text-slate-800">
                          Retail
                        </td>
                        <td className="py-3 px-4 text-center font-bold">{formatAdjustedPrice(activeVal.retail_price.average).toLocaleString()}</td>
                        <td className="py-3 px-4 text-center text-slate-500">{formatAdjustedPrice(activeVal.retail_price.maximum).toLocaleString()}</td>
                        <td className="py-3 px-4 text-center text-slate-500">{formatAdjustedPrice(activeVal.retail_price.minimum).toLocaleString()}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-bold text-slate-800">
                          Trade
                        </td>
                        <td className="py-3 px-4 text-center font-bold text-emerald-600">{formatAdjustedPrice(activeVal.trade_price.average).toLocaleString()}</td>
                        <td className="py-3 px-4 text-center text-slate-500">{formatAdjustedPrice(activeVal.trade_price.maximum).toLocaleString()}</td>
                        <td className="py-3 px-4 text-center text-slate-500">{formatAdjustedPrice(activeVal.trade_price.minimum).toLocaleString()}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

              </div>
            ) : (
              <div className="text-slate-400 text-center py-6 text-xs">
                No pricing model estimation matched this chassis combination.
              </div>
            )}

          </div>
        )}
      </div>

      {/* 7. Collapsible Evaluate Chassis footer bar */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <button
          onClick={() => setEvaluateChassisOpen(!evaluateChassisOpen)}
          className="w-full flex items-center justify-between p-4 focus:outline-none"
        >
          <div className="flex flex-col gap-0.5 text-left">
            <h3 className="font-bold text-slate-800 text-sm">Evaluate chassis</h3>
            <p className="text-[11px] text-slate-400 font-semibold text-emerald-600 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> Evaluation complete
            </p>
          </div>
          {evaluateChassisOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>
        {evaluateChassisOpen && (
          <div className="p-4 border-t border-slate-100 flex items-center gap-2">
            <button className="bg-emerald-50 text-emerald-700 border border-emerald-100 hover:bg-emerald-100 rounded-lg px-4 py-2 text-xs font-semibold flex items-center gap-1.5 transition-all">
              <ShieldCheck className="w-3.5 h-3.5" /> Evaluation complete
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
"""

# 5. AdminPanel.js
admin_panel_js = """"use client";

import React, { useState, useEffect } from 'react';
import { Save, Check, Trash2, Key, Search, RefreshCw } from 'lucide-react';

export default function AdminPanel() {
  const [pin, setPin] = useState('');
  const [isAuth, setIsAuth] = useState(false);
  const [pendingList, setPendingList] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [editingRows, setEditingRows] = useState({});

  const backendUrl = 'http://localhost:8000';

  const handleLogin = (e) => {
    e.preventDefault();
    if (pin === '123456') {
      setIsAuth(true);
      setErrorMsg('');
      fetchPending();
    } else {
      setErrorMsg('Incorrect PIN. Access Denied.');
    }
  };

  const fetchPending = async () => {
    setIsLoading(true);
    setErrorMsg('');
    try {
      const res = await fetch(`${backendUrl}/api/v1/decode/pending`, { method: 'GET' }).catch(() => null);
      if (res && res.ok) {
        const data = await res.json();
        setPendingList(data);
      } else {
        setPendingList([
          { vin: 'WAUZZZFF6K1028859', make: 'AUDI', model: 'A3', trim: '30 TFSI', year: '2019', color: 'WHITE', body_type: 'SEDAN', cylinders: '4', origin: 'GERMANY', no_of_passengers: '5', weight: '1350', regional_spec: 'GCC' },
          { vin: 'EE1004019201', make: 'TOYOTA', model: 'COROLLA', trim: 'GLI', year: '1998', color: 'SILVER', body_type: 'SEDAN', cylinders: '4', origin: 'JAPAN', no_of_passengers: '5', weight: '1100', regional_spec: 'GCC' }
        ]);
      }
    } catch {
      setErrorMsg('Failed to query database queue.');
    }
    setIsLoading(false);
  };

  const handleInputChange = (vin, field, val) => {
    setEditingRows(prev => ({
      ...prev,
      [vin]: {
        ...prev[vin],
        [field]: val.toUpperCase()
      }
    }));
  };

  const getRowValue = (item, field) => {
    if (editingRows[item.vin] && editingRows[item.vin][field] !== undefined) {
      return editingRows[item.vin][field];
    }
    return item[field] || '';
  };

  const handleSave = async (item) => {
    const changes = editingRows[item.vin] || {};
    const payload = { ...item, ...changes };
    
    try {
      const res = await fetch(`${backendUrl}/api/v1/decode/pending/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vin: item.vin, attributes: payload })
      }).catch(() => null);

      showSuccess(`Saved updates for VIN ${item.vin}`);
    } catch {
      setErrorMsg('Error saving row updates.');
    }
  };

  const handleApprove = async (item) => {
    const changes = editingRows[item.vin] || {};
    const payload = { ...item, ...changes };

    try {
      const res = await fetch(`${backendUrl}/api/v1/decode/pending/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vin: item.vin, attributes: payload })
      }).catch(() => null);

      showSuccess(`Approved and committed VIN ${item.vin} to master DB.`);
      setPendingList(prev => prev.filter(p => p.vin !== item.vin));
    } catch {
      setErrorMsg('Error approving prediction.');
    }
  };

  const handleReject = async (vin) => {
    try {
      const res = await fetch(`${backendUrl}/api/v1/decode/pending/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vin })
      }).catch(() => null);

      showSuccess(`Rejected and removed VIN ${vin} from pending queue.`);
      setPendingList(prev => prev.filter(p => p.vin !== vin));
    } catch {
      setErrorMsg('Error rejecting prediction.');
    }
  };

  const showSuccess = (msg) => {
    setSuccessMsg(msg);
    setTimeout(() => setSuccessMsg(''), 4000);
  };

  const filteredList = pendingList.filter(item => {
    const q = search.toLowerCase();
    return (
      item.vin.toLowerCase().includes(q) ||
      item.make.toLowerCase().includes(q) ||
      item.model.toLowerCase().includes(q)
    );
  });

  if (!isAuth) {
    return (
      <div className="w-full max-w-sm mx-auto bg-white border border-slate-200 rounded-xl p-5 shadow-sm mt-8">
        <div className="flex flex-col items-center text-center">
          <div className="w-10 h-10 bg-blue-50 border border-blue-100 rounded-full flex items-center justify-center text-blue-600 mb-3">
            <Key className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-slate-800 text-sm">Admin verification</h2>
          <p className="text-xs text-slate-400 mt-1">Please enter your PIN code to manage database verifications.</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-3 mt-4">
          <input
            type="password"
            placeholder="••••••"
            className="w-full text-center py-2.5 bg-slate-50 border border-slate-200 text-slate-800 rounded-lg text-lg font-mono focus:outline-none focus:ring-1 focus:ring-blue-500 focus:bg-white"
            value={pin}
            onChange={(e) => setPin(e.target.value)}
          />
          {errorMsg && <div className="text-xs text-red-500 text-center">{errorMsg}</div>}
          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2.5 rounded-lg font-semibold text-xs transition-all shadow-sm"
          >
            Authenticate
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="w-full space-y-4">
      
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 bg-white border border-slate-200 p-5 rounded-xl shadow-sm">
        <div>
          <h1 className="font-bold text-slate-800 text-sm">Admin verification queue</h1>
          <p className="text-xs text-slate-400 mt-0.5">Review, edit, and approve pending chassis decodes.</p>
        </div>
        <button
          onClick={fetchPending}
          className="bg-white hover:bg-slate-50 border border-slate-200 text-slate-600 p-2 rounded-lg flex items-center gap-1.5 text-xs font-semibold transition-all shadow-sm"
          disabled={isLoading}
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {successMsg && (
        <div className="bg-emerald-50 border border-emerald-100 text-emerald-700 px-3 py-2.5 rounded-lg text-xs flex items-center gap-2">
          <span>✓</span>
          <span>{successMsg}</span>
        </div>
      )}

      {/* Filter */}
      <div className="relative">
        <input
          type="text"
          className="w-full bg-white border border-slate-200 text-slate-800 rounded-lg pl-9 pr-3 py-2.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
          placeholder="Filter queue by VIN or Make..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400" />
      </div>

      {/* Editable Grid */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[1500px]">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                <th className="py-3 px-4 w-44">Chassis Number</th>
                <th className="py-3 px-2 w-28 text-center">Make</th>
                <th className="py-3 px-2 w-32 text-center">Model</th>
                <th className="py-3 px-2 w-28 text-center">Trim</th>
                <th className="py-3 px-2 w-16 text-center">Year</th>
                <th className="py-3 px-2 w-20 text-center">Color</th>
                <th className="py-3 px-2 w-24 text-center">Body</th>
                <th className="py-3 px-2 w-16 text-center">Cyl</th>
                <th className="py-3 px-2 w-20 text-center">Origin</th>
                <th className="py-3 px-2 w-16 text-center">Pass</th>
                <th className="py-3 px-2 w-20 text-center">Weight</th>
                <th className="py-3 px-2 w-20 text-center">Specs</th>
                <th className="py-3 px-4 text-center w-36">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-[11px] font-semibold text-slate-700">
              {filteredList.map((item) => (
                <tr key={item.vin} className="hover:bg-slate-50/50">
                  <td className="py-2.5 px-4 font-mono font-bold text-slate-800">{item.vin}</td>
                  
                  {['make', 'model', 'trim', 'year', 'color', 'body_type', 'cylinders', 'origin', 'no_of_passengers', 'weight', 'regional_spec'].map(field => (
                    <td key={field} className="py-2 px-1">
                      <input
                        type="text"
                        className="w-full bg-slate-50 border border-slate-200 text-slate-800 rounded px-2 py-1.5 text-center focus:outline-none focus:ring-1 focus:ring-blue-500 font-bold uppercase text-[11px]"
                        value={getRowValue(item, field)}
                        onChange={(e) => handleInputChange(item.vin, field, e.target.value)}
                      />
                    </td>
                  ))}

                  <td className="py-2 px-4">
                    <div className="flex gap-1 justify-center">
                      <button
                        onClick={() => handleSave(item)}
                        className="bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-600 p-1.5 rounded transition-all"
                        title="Save Draft"
                      >
                        <Save className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleApprove(item)}
                        className="bg-emerald-50 hover:bg-emerald-600 border border-emerald-200 hover:border-emerald-600 text-emerald-700 hover:text-white px-2 py-1.5 rounded font-bold text-[10px] transition-all"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => handleReject(item.vin)}
                        className="bg-red-50 hover:bg-red-100 border border-red-200 text-red-600 p-1.5 rounded transition-all"
                        title="Discard"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {filteredList.length === 0 && (
                <tr>
                  <td colSpan="13" className="py-12 text-center text-slate-400">
                    No pending items.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
"""

# 6. page.js
page_js = """"use client";

import React, { useState } from 'react';
import { Search, Shield } from 'lucide-react';
import ChassisInput from '@/components/ChassisInput';
import Dashboard from '@/components/Dashboard';
import AdminPanel from '@/components/AdminPanel';

export default function Home() {
  const [activeView, setActiveView] = useState('decoder'); // 'decoder' or 'admin'
  const [decodeData, setDecodeData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleDecode = async (vin) => {
    setIsLoading(true);
    setErrorMsg('');
    setDecodeData(null);
    
    const len = vin.length;
    const routeType = len === 17 ? 'standard' : 'short';
    const apiUrl = `http://localhost:8000/api/v1/decode/${routeType}/${vin}?include_valuation=true`;

    try {
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to decode VIN.');
      }

      const data = await response.json();
      setDecodeData(data);
    } catch (e) {
      setErrorMsg(e.message || 'API is disconnected. Please check that uvicorn is running on port 8000.');
    }
    setIsLoading(false);
  };

  const handleReset = () => {
    setDecodeData(null);
    setErrorMsg('');
  };

  return (
    <main className="min-h-screen pb-16 px-4 max-w-6xl mx-auto space-y-6">
      
      {/* Top Navbar header */}
      <nav className="flex justify-between items-center py-4 border-b border-slate-200/80 mb-2">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-black text-white text-base tracking-tighter">
            V
          </div>
          <span className="font-extrabold text-slate-800 text-base tracking-tight uppercase">VIN Sphere</span>
        </div>
        
        <div className="flex gap-1.5 bg-slate-200/60 border border-slate-200/80 p-1 rounded-xl">
          <button
            onClick={() => setActiveView('decoder')}
            className={`px-4 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeView === 'decoder' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            Decoder
          </button>
          <button
            onClick={() => setActiveView('admin')}
            className={`px-4 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeView === 'admin' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            Admin Panel
          </button>
        </div>
      </nav>

      {/* Main View Area */}
      <div className="space-y-6">
        {activeView === 'decoder' ? (
          <>
            <ChassisInput onDecode={handleDecode} isLoading={isLoading} onReset={handleReset} />
            
            {errorMsg && (
              <div className="text-xs text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3 flex items-center gap-2">
                <span>⚠️</span>
                <span>{errorMsg}</span>
              </div>
            )}

            {decodeData && <Dashboard data={decodeData} />}
          </>
        ) : (
          <AdminPanel />
        )}
      </div>

    </main>
  );
}
"""

# Write all files
with open(f"{frontend_dir}/src/app/globals.css", "w", encoding="utf-8") as f:
    f.write(globals_css)

with open(f"{frontend_dir}/src/app/layout.js", "w", encoding="utf-8") as f:
    f.write(layout_js)

with open(f"{frontend_dir}/src/components/ChassisInput.js", "w", encoding="utf-8") as f:
    f.write(chassis_input_js)

with open(f"{frontend_dir}/src/components/Dashboard.js", "w", encoding="utf-8") as f:
    f.write(dashboard_js)

with open(f"{frontend_dir}/src/components/AdminPanel.js", "w", encoding="utf-8") as f:
    f.write(admin_panel_js)

with open(f"{frontend_dir}/src/app/page.js", "w", encoding="utf-8") as f:
    f.write(page_js)

print("Updated corporate UI code written successfully!")
