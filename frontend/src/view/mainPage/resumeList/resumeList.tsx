import axiosInstance from "@/api/api";
import { Button, Form, Pagination } from "antd";
import "./resumeList.css";
import { EyeOutlined, StarOutlined } from "@ant-design/icons";
import { useEffect, useState } from "react";
import Search from "antd/es/input/Search";
import { Link } from "react-router-dom";
import React from "react";
import axios from "axios";

interface ResumeList {
  num: number;
  r_num: number;
  rating: number;
  rating_count: number;
  read_num: number;
  title: string;
  w_date: string;
  content: string;
}

const ResumeList = () => {
  const [resumeList, setResumeList] = useState<ResumeList[]>([]);
  const [topRatedResumes, setTopRatedResumes] = useState<ResumeList[]>([]);
  const [topReadResumes, setTopReadResumes] = useState<ResumeList[]>([]);
  const [totalPages, setTotalPages] = useState<number>(0);

  async function fetchSearch(search: string) {
    let res = await axiosInstance.get("/board/list/search", {
      params: {
        pageNo: 0,
        keyword: search,
      },
    });
    setResumeList(res.data.response);
    setTotalPages(res.data.totalPages);
  }

  async function fetchList(page: number) {
    let res = await axiosInstance.get("/board/list", {
      params: {
        pageNo: page - 1,
      },
    });
    setResumeList(res.data.response);
    setTotalPages(res.data.totalPages);
  }

  async function fetchTopResumes(
    group: string,
    setter: (data: ResumeList[]) => void
  ) {
    let res = await axiosInstance.get("/board/list/rank", {
      params: { group },
    });
    setter(res.data.response);
  }

  useEffect(() => {
    fetchList(0);
    fetchTopResumes("rating", setTopRatedResumes);
    fetchTopResumes("read_num", setTopReadResumes);
  }, []);

  return (
    <div style={{ display: "flex" }}>
      <div
        className="leftWrapper"
        style={{
          width: "30%",
          padding: "1%",
          marginLeft: "5%",
          marginTop: "10.2%",
        }}
      >
        <div
          style={{
            backgroundColor: "#85dad2",
            padding: "1px",
            borderRadius: "4px",
            textAlign: "center",
            marginBottom: "20px",
            color: "white",
          }}
        >
          <h2>Top Rated Resumes</h2>
        </div>
        {topRatedResumes.map((resume, index) => (
          <div
            key={`topRated${index}`}
            className="contentBox"
            style={{
              border: "1px solid rgb(225,225,225)",
              padding: "2% 5%",
              margin: "1% 0",
              position: "relative",
              display: "flex",
            }}
          >
            <img
              src="/img/recommended.webp"
              alt="Resume"
              style={{
                width: "60px",
                height: "60px",
                marginRight: "20px",
                marginTop: "30px",
              }}
            />
            <div style={{ flex: "1" }}>
              <h4 style={{ color: "black", marginBottom: "4px" }}>
                <Link
                  to={`/main/resumelist/${resume.r_num}`}
                  style={{ color: "black" }}
                >
                  {resume.title}
                </Link>
              </h4>
              <p
                style={{
                  marginTop: "4px",
                  fontSize: "13px",
                  display: "-webkit-box",
                  WebkitBoxOrient: "vertical",
                  WebkitLineClamp: 2,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {resume.content}
              </p>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span>{resume.w_date}</span>
                <span>
                  <EyeOutlined /> {resume.read_num}
                </span>
              </div>
              <div style={{ position: "absolute", top: "10px", right: "10px" }}>
                <StarOutlined /> {resume.rating.toFixed(1)}
              </div>
            </div>
          </div>
        ))}

        <div
          style={{
            backgroundColor: "#85dad2",
            padding: "1px",
            borderRadius: "4px",
            textAlign: "center",
            marginTop: "50px",
            marginBottom: "20px",
            color: "white",
          }}
        >
          <h2>Most Viewed Resumes</h2>
        </div>
        {topReadResumes.map((resume, index) => (
          <div
            key={`topRead${index}`}
            className="contentBox"
            style={{
              border: "1px solid rgb(225,225,225)",
              padding: "2% 5%",
              margin: "1% 0",
              position: "relative",
              display: "flex",
            }}
          >
            <img
              src="/img/most_viewed.webp"
              alt="Resume"
              style={{
                width: "60px",
                height: "60px",
                marginRight: "20px",
                marginTop: "30px",
              }}
            />
            <div style={{ flex: "1" }}>
              <h4 style={{ color: "black", marginBottom: "4px" }}>
                <Link
                  to={`/main/resumelist/${resume.r_num}`}
                  style={{ color: "black" }}
                >
                  {resume.title}
                </Link>
              </h4>
              <p
                style={{
                  fontSize: "13px",
                  marginTop: "4px",
                  display: "-webkit-box",
                  WebkitBoxOrient: "vertical",
                  WebkitLineClamp: 2,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {resume.content}
              </p>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span>{resume.w_date}</span>
                <span>
                  <EyeOutlined /> {resume.read_num}
                </span>
              </div>
              <div style={{ position: "absolute", top: "10px", right: "10px" }}>
                <StarOutlined /> {resume.rating.toFixed(1)}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rightWrapper" style={{ width: "75%", padding: "1%" }}>
        <div
          className="search"
          style={{
            display: "flex",
            justifyContent: "center",
            paddingTop: "3%",
          }}
        >
          <Search
            onSearch={(value) => {
              fetchSearch(value);
            }}
            size="large"
            placeholder="검색어를 입력하세요"
            style={{ width: "60%" }}
          />
        </div>
        <div
          style={{ display: "flex", justifyContent: "center", margin: "3% 0" }}
        >
          <Pagination
            onChange={(page) => {
              fetchList(page);
            }}
            pageSize={5}
            showSizeChanger={false}
            total={totalPages * 5 - 5}
          />
        </div>
        <div className="wrapper">
          <div
            className="innerWrapper"
            style={{ display: "flex", width: "100%", flexDirection: "column" }}
          >
            {resumeList.map((resume, index) => (
              <div
                key={`resume.r_num${index}`}
                style={{ display: "flex", justifyContent: "center" }}
              >
                <div
                  className="contentBox"
                  style={{
                    width: "80%",
                    border: "1px solid rgb(225,225,225)",
                    padding: "2% 5%",
                    margin: "1% 0",
                  }}
                >
                  <div className="contentTitle">
                    <h3>
                      <Link
                        style={{ color: "black" }}
                        to={`/main/resumelist/${resume.r_num}`}
                      >
                        {resume.title}
                      </Link>
                    </h3>
                  </div>
                  <div
                    className="contentMainText"
                    style={{
                      display: "-webkit-box",
                      WebkitBoxOrient: "vertical",
                      margin: "1% 0",
                      fontSize: "16px",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      WebkitLineClamp: 2,
                      width: "100%",
                    }}
                  >
                    {resume.content}
                  </div>
                  <div
                    className="contentFooter"
                    style={{
                      justifyContent: "space-between",
                      display: "flex",
                      fontSize: "14px",
                    }}
                  >
                    <div className="contentDate">{resume.w_date}</div>
                    <div className="contentView">
                      <EyeOutlined /> {resume.read_num}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResumeList;
