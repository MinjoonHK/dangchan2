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

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  let accessToken = localStorage.getItem("access") ?? "";
  let decodedToken: DecodedToken | null = null;

  try {
    decodedToken = jwtDecode<DecodedToken>(accessToken);
  } catch (error) {
    console.error("Failed to decode token", error);
    return <Navigate to="/" replace />;
  }

  if (!decodedToken || decodedToken.role !== "ROLE_ADMIN") {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
