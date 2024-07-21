import MainPage from "@/view/mainPage/mainPage";
import MyPage from "@/view/mainPage/myPage/myPage";
import ResumeEdit from "@/view/mainPage/resumeEdit/resumeEdit";
import ResumeGuide from "@/view/mainPage/resumeGuide/resumeGuide";
import ResumeListDetails from "@/view/mainPage/resumeList/details/details";
import ResumeHistoryDetail from "@/view/mainPage/resumeList/details/resumeHistory";
import ResumeList from "@/view/mainPage/resumeList/resumeList";
import { Route, Routes } from "react-router-dom";

const MainRoutes = () => (
  <Routes>
    <Route path="/main" element={<MainPage />}>
      <Route path="resume" element={<ResumeEdit />} />
      <Route path="resumeguide" element={<ResumeGuide />} />
      <Route path="resumelist/:id" element={<ResumeListDetails />} />
      <Route path="resumelist" element={<ResumeList />} />
      <Route path="mypage" element={<MyPage />} />
      <Route path="mypage/:id" element={<ResumeHistoryDetail />} />
    </Route>
  </Routes>
);

export default MainRoutes;
