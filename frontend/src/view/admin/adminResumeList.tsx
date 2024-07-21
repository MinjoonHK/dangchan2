import axiosInstance from "@/api/api";
import Table from "antd/es/table";
import React, { useEffect, useState } from "react";

interface ResumeList {
  page: number;
  rating: number;
  rating_count: number;
  read_num: number;
  rnum: number;
  size: number;
  title: string;
}

const tableColumns = [
  {
    title: "제목",
    dataIndex: "title",
  },
  {
    title: "평점",
    dataIndex: "rating",
    sorter: (a, b) => a.rating - b.rating,
  },
  {
    title: "조회수",
    dataIndex: "read_num",
    sorter: (a, b) => a.read_num - b.read_num,
  },
  {
    title: "사용자",
    dataIndex: "username",
  },
  {
    title: "작성날짜",
    dataIndex: "w_date",
    sorter: (a, b) =>
      new Date(a.w_date).getTime() - new Date(b.w_date).getTime(),
  },
];
const AdminResumeList = () => {
  const [resumeList, setResumeList] = useState<ResumeList[]>([]);
  const [totalPage, setTotalPage] = useState<number>(0);
  const fetchResumeList = (page: number) => {
    let res = axiosInstance
      .get("/admin/resume/board", {
        params: {
          pageNo: page,
        },
      })
      .then((res) => {
        console.log(res);
        let newData = res.data.resume_board_list.map((data: any, idx) => {
          return {
            ...data,
            key: idx,
            w_date: data.w_date.slice(0, 10),
          };
        });
        setResumeList(newData);
        setTotalPage(res.data.total_page);
      });
  };
  useEffect(() => {
    fetchResumeList(0);
  }, []);
  return (
    <div style={{ padding: "2% 5%" }}>
      <Table
        pagination={{
          showSizeChanger: false,
          onChange: (page) => {
            fetchResumeList(page - 1);
          },
          pageSize: 10,
          total: totalPage * 10,
        }}
        dataSource={resumeList}
        columns={tableColumns}
      />
    </div>
  );
};

export default AdminResumeList;
