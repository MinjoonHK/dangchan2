import { Route, Routes } from "react-router-dom";
import ProtectedRoute from "./protectedRoutes";
import AdminPage from "@/view/admin/adminPage";

const AdminRoutes = () => (
  <Routes>
    <Route
      path="/admin/main"
      element={
        <ProtectedRoute>
          <AdminPage />
        </ProtectedRoute>
      }
    />
  </Routes>
);

export default AdminRoutes;
