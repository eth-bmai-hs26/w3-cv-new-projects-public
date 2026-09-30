import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./styles.css";
import { Shell } from "./components/Shell";
import { InspectorProvider, ModelProvider, ToastProvider } from "./lib/hooks";
import Dashboard from "./pages/Dashboard";
import Inspect from "./pages/Inspect";
import { LotDetailPage, LotReportPage, LotsPage } from "./pages/Lots";
import Review from "./pages/Review";
import Settings from "./pages/Settings";

function NotFound() {
  return (
    <div className="page">
      <h1>Page not found</h1>
      <p><a href="/">Go to the dashboard</a></p>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <InspectorProvider>
        <ModelProvider>
          <ToastProvider>
            <Routes>
              <Route path="/lots/:id/report" element={<LotReportPage />} />
              <Route element={<Shell />}>
                <Route index element={<Dashboard />} />
                <Route path="inspect" element={<Inspect />} />
                <Route path="lots" element={<LotsPage />} />
                <Route path="lots/:id" element={<LotDetailPage />} />
                <Route path="review" element={<Review />} />
                <Route path="settings" element={<Settings />} />
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </ToastProvider>
        </ModelProvider>
      </InspectorProvider>
    </BrowserRouter>
  </StrictMode>,
);
