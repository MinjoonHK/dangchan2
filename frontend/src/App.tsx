import React, { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import ProtectedRoute from "./routes/protectedRoutes";
import ProtectLoginRoute from "./routes/protectLoginRoute";
import LandingPage from "./view/landingPage/landing";
// lazy하게 로드할 컴포넌트들
const Login = lazy(() => import("./view/auth/login"));
const SignUp = lazy(() => import("./view/auth/signUp"));
const NotFound = lazy(() => import("./view/error/notFound/notFound"));
const AdminPage = lazy(() => import("./view/admin/adminPage"));
const MainPage = lazy(() => import("./view/mainPage/mainPage"));
import "./App.css";
function App() {
  return (
    <div className="App" style={{ margin: "0" }}>
      <Suspense fallback={<></>}>
        <Routes>
          <Route path="/auth/signup" element={<SignUp />} />
          <Route path="/auth/login" element={<Login />} />
          <Route
            path="/"
            element={
              <ProtectLoginRoute>
                <LandingPage />
              </ProtectLoginRoute>
            }
          />
          <Route path="/main/*" element={<MainPage />} />
          <Route
            path="/admin/main"
            element={
              <ProtectedRoute>
                <AdminPage />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </div>
  );
}

export default App;
