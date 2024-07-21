import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axiosInstance from "@/api/api";
import { jwtDecode } from "jwt-decode";
import { DecodedToken } from "@/types/globalTypes";
import React from "react";

interface Resume {
  mode: number;
  occupation: string;
  r_num: number;
  company: string;
  r_content: string;
  w_date: string;
  item: string;
  content: string;
}

const ResumeHistoryDetail = () => {
  const [resume, setResume] = useState<Partial<Resume>>({});
  const [isLoading, setIsLoading] = useState(false);

  const accessToken = localStorage.getItem("access") ?? "";
  let DecodedToken: DecodedToken = jwtDecode(accessToken);
  const userInfo = DecodedToken.username;
  const param = useParams();
  const fetchResume = () => {
    let res = axiosInstance.get(`/user/edit-list/${param.id}`).then((res) => {
      setResume(res.data);
    });
  };
  useEffect(() => {
    let res = Promise.all([fetchResume()]);
  }, []);
  return (
    <div className="Wrapper" style={{ padding: "5% 5%", display: "flex" }}>
      <div
        className="userInnerWrapper"
        style={{
          border: "1px solid rgb(220,220,220)",
          boxShadow: "0 0 10px 0 rgb(220, 220, 220)",
          borderRadius: "5px",
          height: "100%",
          width: "50%",
          padding: "5%",
        }}
      >
        <div>
          <div>{resume.w_date}</div>
          <div
            style={{
              fontSize: "1.2rem",
              marginTop: "15px",
              marginBottom: "15px",
            }}
          >
            {resume.company} {resume.occupation}
          </div>
          <div>{resume.r_content}</div>
          <div style={{ marginTop: "15px" }}>
            글자수: {resume.r_content?.length}
          </div>
        </div>
      </div>
      <div
        className="gptInnerWrapper"
        style={{
          border: "1px solid rgb(220,220,220)",
          boxShadow: "0 0 10px 0 rgb(220, 220, 220)",
          borderRadius: "5px",
          height: "100%",
          width: "50%",
          marginLeft: "4%",
          padding: "5%",
        }}
      >
        <div>
          <h3>수정된 자소서</h3>
          <div>
            <div
              style={{
                whiteSpace: "pre-wrap",
                fontWeight: "bold",
                fontSize: "1rem",
              }}
            >
              {resume.content}
            </div>
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginTop: "10px",
            }}
          >
            글자수: {resume.content?.length}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResumeHistoryDetail;
