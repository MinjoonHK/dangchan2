# import csv 
# import glob
# import os
# import pdfkit

# f = open("/mnt/d/dacon_chatbot/bozo_data/serviceDetail.csv","r")
# reader = csv.DictReader(f)

# def html2pdf(file_path):
#     try:
#         # wkhtmltopdf 경로 설정
#         path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # wkhtmltopdf의 경로로 변경
#         config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        
#         # HTML 파일을 PDF로 변환
#         output_file = file_path.replace('.html', '.pdf')
#         pdfkit.from_file(file_path, output_file, configuration=config)
        
#         return True
#     except Exception as err:
#         print(f"Error: {err}")
#         return False

# if __name__ == "__main__":
#     path = os.path.abspath("first_row.html")
#     html2pdf(path)

# for row in reader:
#     print(row)
#     html_content = "<html>\n<head>\n<title>First Row</title>\n<meta charset='UTF-8'>\n<style>\n"
#     html_content += "body { font-family: 'Nanum Gothic', sans-serif; }\n"
#     html_content += "</style>\n</head>\n<body>\n<table border='1'>\n"
#     # 테이블 헤더 추가
#     # html_content += "<tr>\n"
#     # for key in row.keys():
#     #     html_content += f"<th>{key}</th>\n"
#     # html_content += "</tr>\n"

#     # 테이블 데이터 추가 (행과 열을 바꾸기)
#     for key, value in row.items():
#         html_content += "<tr>\n"
#         html_content += f"<th>{key}</th>\n"
#         html_content += f"<td>{value}</td>\n"
#         html_content += "</tr>\n"

#     html_content += "</table>\n</body>\n</html>"
    
#     # HTML 파일로 저장
#     with open("first_row.html", "w", encoding='utf-8') as file:
#         file.write(html_content)
#     break

# f.close()

import csv
import os
import pdfkit
import PyPDF2
import pikepdf
import io
from tqdm import tqdm

# CSV 파일 경로
csv_file_path = "/mnt/d/dacon_chatbot/bozo_data/serviceDetail.csv"

# HTML을 PDF로 변환하는 함수 (메모리에서)
def html2pdf(html_content):
    try:
        # wkhtmltopdf 경로 설정
        path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # wkhtmltopdf의 경로로 변경
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        
        # HTML을 PDF로 변환 (메모리에서)
        pdf_output = pdfkit.from_string(html_content, False, configuration=config)
        
        return pdf_output
    except Exception as err:
        print(f"Error: {err}")
        return None

# 모든 PDF 파일을 하나로 합치는 함수 (메모리에서)
def merge_pdfs(pdf_files, output_path):
    try:
        pdf_writer = pikepdf.Pdf.new()
        
        for pdf_data in pdf_files:
            pdf_reader = pikepdf.Pdf.open(io.BytesIO(pdf_data))
            pdf_writer.pages.extend(pdf_reader.pages)

        pdf_writer.save(output_path)
        print(f"PDFs merged successfully into {output_path}")
        return True
    except Exception as err:
        print(f"Error: {err}")
        return False

if __name__ == "__main__":
    pdf_files = []
    
    # CSV 파일 열기
    with open(csv_file_path, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # 모든 행을 반복
        for i, row in tqdm(enumerate(rows), total=len(rows), desc="Processing rows"):
            # HTML 콘텐츠 생성
            html_content = "<html>\n<head>\n<title>Row {i+1}</title>\n<meta charset='UTF-8'>\n<style>\n"
            html_content += "body { font-family: 'Nanum Gothic', sans-serif; }\n"
            html_content += "</style>\n</head>\n<body>\n<table border='1'>\n"

            # # 테이블 헤더 추가
            # html_content += "<tr>\n"
            # for key in row.keys():
            #     html_content += f"<th>{key}</th>\n"
            # html_content += "</tr>\n"

            # 테이블 데이터 추가 (행과 열을 바꾸기)
            for key, value in row.items():
                html_content += "<tr>\n"
                html_content += f"<th>{key}</th>\n"
                html_content += f"<td>{value}</td>\n"
                html_content += "</tr>\n"

            # # HTML 파일로 저장
            # html_file_path = f"row_{i+1}.html"
            # with open(html_file_path, "w", encoding='utf-8') as file:
            #     file.write(html_content)

            # HTML 파일을 PDF로 변환
            pdf_file = html2pdf(html_content)
            if pdf_file:
                pdf_files.append(pdf_file)
    

        # # 첫 번째 행 읽기
        # first_row = next(reader)

        # # HTML 콘텐츠 생성
        # html_content = "<html>\n<head>\n<title>First Row</title>\n<meta charset='UTF-8'>\n<style>\n"
        # html_content += "body { font-family: 'Nanum Gothic', sans-serif; }\n"
        # html_content += "</style>\n</head>\n<body>\n<table border='1'>\n"

        # # # 테이블 헤더 추가
        # # html_content += "<tr>\n"
        # # for key in first_row.keys():
        # #     html_content += f"<th>{key}</th>\n"
        # # html_content += "</tr>\n"

        # # 테이블 데이터 추가 (행과 열을 바꾸기)
        # for key, value in first_row.items():
        #     html_content += "<tr>\n"
        #     html_content += f"<th>{key}</th>\n"
        #     html_content += f"<td>{value}</td>\n"
        #     html_content += "</tr>\n"

        # # HTML 파일로 저장
        # html_file_path = "first_row.html"
        # with open(html_file_path, "w", encoding='utf-8') as file:
        #     file.write(html_content)

        # # HTML 파일을 PDF로 변환
        # pdf_file = html2pdf(html_file_path)
        # if pdf_file:
        #     pdf_files.append(pdf_file)
    
    # 모든 PDF 파일을 하나로 합치기
    output_pdf_path = "combined.pdf"
    merge_pdfs(pdf_files, output_pdf_path)
