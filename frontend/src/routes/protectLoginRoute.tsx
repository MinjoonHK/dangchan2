import { jwtDecode } from "jwt-decode";
import { Navigate } from "react-router-dom";
import React from "react";

interface DecodedToken {
  category: string;
  username: string;
  role: string;
  uNum: number;
  iat: number;
  exp: number;
}
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectLoginRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  let accessToken = localStorage.getItem("access") ?? "";
  let decodedToken: DecodedToken | null = null;

  if (accessToken !== "") {
    return <Navigate to="/main/resume" replace />;
  }

  return <>{children}</>;
};

export default ProtectLoginRoute;
