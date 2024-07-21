import Login from "@/view/auth/login";
import SignUp from "@/view/auth/signUp";
import React from "react";
import { Route, Routes } from "react-router-dom";

const AuthRoutes = () => (
  <>
    <React.Suspense fallback={<div>Loading...</div>}>
      <Route path="/auth/signup" element={<SignUp />} />
      <Route path="/auth/login" element={<Login />} />
    </React.Suspense>
  </>
);

export default AuthRoutes;
