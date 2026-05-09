"use client";

import React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

/**
 * ErrorBoundary — catches React render errors in child trees.
 *
 * Usage:
 *   <ErrorBoundary label="MLOps Dashboard">
 *     <MLOpsDashboard />
 *   </ErrorBoundary>
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      const { label = "Component" } = this.props;
      return (
        <div className="flex-1 flex items-center justify-center p-10 bg-[#f8f9ff] min-h-[400px]">
          <div className="bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.06)] p-10 max-w-md w-full text-center">
            <div className="w-14 h-14 rounded-2xl bg-red-50 border border-red-100 flex items-center justify-center mx-auto mb-5">
              <AlertTriangle className="w-7 h-7 text-[#ba1a1a]" />
            </div>
            <h3 className="text-[17px] font-bold text-[#0b1c30] mb-2">
              {label} crashed
            </h3>
            <p className="text-[13px] text-[#777682] mb-1 leading-relaxed">
              An unexpected error occurred while rendering this component.
            </p>
            {this.state.error && (
              <pre className="mt-3 mb-5 text-left text-[10px] font-mono bg-gray-50 border border-gray-200 rounded-xl p-3 overflow-auto max-h-32 text-red-700">
                {this.state.error.message}
              </pre>
            )}
            <button
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#000018] text-white
                text-[13px] font-semibold hover:bg-[#000018]/85 active:scale-[0.98]
                transition-all duration-150 cursor-pointer"
            >
              <RefreshCw className="w-4 h-4" />
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
