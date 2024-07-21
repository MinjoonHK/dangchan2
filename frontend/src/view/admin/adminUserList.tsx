import axiosInstance from "@/api/api";
import Button from "antd/es/button";
import Input from "antd/es/input";
import Select from "antd/es/select";
import Space from "antd/es/space";
import Table from "antd/es/table";
import React, { useEffect, useState } from "react";
import Swal from "sweetalert2";
import { Modal } from "antd";

const { Search } = Input;

interface UserList {
  age: number;
  authCode: string;
  birthDate: string;
  company: string | null;
  delDate: string;
  email: string;
  gender: string;
  inDate: string;
  mode: number;
  occupation: string | null;
  password: null;
  resumeEditCount: number;
  role: string;
  status: number;
  unum: number;
  username: string;
  wish: string | null;
}
const dropDownOptions = [
  {
    value: "username",
    label: "유저아이디",
  },
  {
    value: "email",
    label: "이메일",
  },
  {
    value: "company",
    label: "회사",
  },
  {
    value: "occupation",
    label: "직업",
  },
  {
    value: "wish",
    label: "희망직종",
  },
];

const AdminUserList = () => {
  const [userList, setUserList] = useState<UserList[]>([]);
  const [totalPage, setTotalPage] = useState<number>(0);
  const [searchOption, setSearchOption] = useState("");
  const [searchText, setSearchText] = useState("");
  const [searchedText, setSearchedText] = useState<string>("");
  const [selectedRow, setSelectedRow] = useState<Partial<UserList>>({});
  const [openDeleteModal, setOpenDeleteModal] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState<number>(1);

  const tableColumns = [
    {
      title: "아이디",
      dataIndex: "username",
    },
    {
      title: "이메일",
      dataIndex: "email",
    },
    {
      title: "성별",
      dataIndex: "gender",
      filters: [
        {
          text: "남성",
          value: "M",
        },
        {
          text: "여성",
          value: "F",
        },
      ],
      onFilter: (value, record) => record.gender.indexOf(value as string) === 0,
    },
    {
      title: "유형",
      dataIndex: "role",
    },
    {
      title: "가입일",
      dataIndex: "inDate",
      sorter: (a, b) =>
        new Date(a.inDate).getTime() - new Date(b.inDate).getTime(),
    },
    {
      title: "삭제일",
      dataIndex: "delDate",
      sorter: (a, b) =>
        new Date(a.delDate).getTime() - new Date(b.delDate).getTime(),
    },
    {
      title: "Action",
      dataIndex: "",
      render: () => (
        <Button
          onClick={() => {
            setOpenDeleteModal(true);
          }}
        >
          블랙리스트
        </Button>
      ),
    },
  ];
  const fetchSearch = (page: number, search: string) => {
    let res = axiosInstance
      .get("/admin/user/list/search", {
        params: {
          pageNo: page,
          keyword: search,
          group: searchOption,
        },
      })
      .then((res) => {
        let newData = res.data.user_list.map((data: any, idx: number) => {
          return {
            ...data,
            key: idx,
            inDate: data.inDate.slice(0, 10),
            delDate: data.delDate ? data.delDate.slice(0, 10) : null,
          };
        });
        setUserList(newData);
        setTotalPage(res.data.total_page);
      })
      .catch((err) => {
        Swal.fire({
          icon: "error",
          title: "검색 결과가 없습니다",
          text: "다시 시도해주세요",
        });
      });
  };
  const onSearch = (search: string) => {
    if (searchOption === "") {
      alert("검색 옵션을 선택해주세요");
      return;
    } else {
      setCurrentPage(1);
      setSearchedText(search);
      fetchSearch(0, search);
    }
  };
  const fetchUserList = (page: number) => {
    let res = axiosInstance
      .get("/admin/user/list", {
        params: {
          pageNo: page,
        },
      })
      .then((res) => {
        console.log(res);
        let newData = res.data.user_list.map((data: any, idx: number) => {
          return {
            ...data,
            key: idx,
            inDate: data.inDate.slice(0, 10),
            delDate: data.delDate ? data.delDate.slice(0, 10) : null,
          };
        });
        setUserList(newData);
        setTotalPage(res.data.total_page);
      });
  };
  useEffect(() => {
    fetchUserList(0);
  }, []);
  return (
    <div style={{ padding: "2% 5%" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          marginBottom: "3%",
        }}
      >
        <Space.Compact style={{ width: "50%" }}>
          <Select
            onChange={(e) => setSearchOption(e as string)}
            size="large"
            defaultValue="검색 옵션"
            options={dropDownOptions}
          />
          <Search
            onChange={(e) => setSearchText(e.target.value)}
            size="large"
            onSearch={onSearch}
          />
        </Space.Compact>
      </div>
      <Table
        onRow={(r) => {
          return {
            onClick: () => {
              setSelectedRow(r);
            },
          };
        }}
        pagination={{
          showSizeChanger: false,
          current: currentPage,
          onChange: (page) => {
            setCurrentPage(page);
            if (searchOption === "") fetchUserList(page - 1);
            else {
              fetchSearch(page - 1, searchText);
            }
          },
          pageSize: 20,
          total: totalPage * 20,
        }}
        dataSource={userList}
        columns={tableColumns}
      />
      <Modal
        centered
        closeIcon={false}
        open={openDeleteModal}
        footer={[
          <Button
            style={{ backgroundColor: "#001529", color: "white" }}
            key={"ok"}
            onClick={() => {
              let res = axiosInstance
                .delete(`/admin/user/${selectedRow.unum}`)
                .then((res) => {
                  if (searchedText === "") fetchUserList(currentPage - 1);
                  else {
                    fetchSearch(currentPage - 1, searchText);
                  }
                })
                .catch((err) => {
                  if (err.response.status === 400) {
                    Swal.fire({
                      icon: "error",
                      title: "삭제 실패",
                      text: "이미 삭제된 유저입니다",
                    });
                  }
                })
                .finally(() => {
                  setOpenDeleteModal(false);
                });
            }}
          >
            예
          </Button>,
          <Button
            key={"no"}
            onClick={() => {
              setOpenDeleteModal(false);
            }}
          >
            아니오
          </Button>,
        ]}
      >
        정말 {selectedRow.username} 유저를 블랙리스트 처리 하시겠습니까?
      </Modal>
    </div>
  );
};

export default AdminUserList;
