#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "csvparser.h"
#include <ctime>

using namespace std;
using namespace cv;

/** Global variables */
String face_cascade_name;
CascadeClassifier face_cascade;
const int
num_of_files = 1, //кол-во файлов для подсчета от 1 до 10;
num_of_steps = 1, // кол-во шагов
num_of_neigh = 1; // кол-во соседей
const float
eps1 = 1,
eps2 = 1,
eps3 = 1,
eps4 = 1;
// счетчики
int summ_of_img, summ_of_all_faces, summ_of_all_frames, TP, TN, FP, DB, time_;
//
const double step[num_of_steps] = { 1.2 };
const int neigh[num_of_neigh] = { 1 };
fstream res;
/** @function main */
int main(int argc, const char* argv[])
{

	setlocale(LC_ALL, "Russian");
	CommandLineParser parser(argc, argv,
		"{face_cascade|C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml|}");
	face_cascade_name = parser.get<string>("face_cascade");
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); cerr << -1; }; // Load the cascades
	fstream read1;// файл для чтения
	string str; // хранение адреса картинки, считанной из файла
	//remove("res.csv");
	res.open("res_default.csv", ios::app);
	res << "eps" << ";" << "step" << ";" << "neigh" << ";" << "TP" << ";" << "FP" << ";" << "TN" << ";" << "DB" << ";" << "summ_of_img" << ";" << "summ_of_faces" << ";" << "summ_of_frames" << ";" << "time/img(ms)" << endl;
	res.close();
	for (int l = 0; l < num_of_neigh; l++)
	{
		for (int p = 0; p < num_of_steps; p++)
		{
			summ_of_img = 0;
			summ_of_all_faces = 0;
			summ_of_all_frames = 0;
			TP = 0;
			TN = 0;
			FP = 0;
			DB = 0;
			time_ = 0;
			for (int i = 0; i < num_of_files; i++)
			{
				read1.open("C:/Users/Анатолий Долгих/source/repos/CV/Проект4/FDDB-fold-0" + to_string(i + 1) + "-ellipseList.txt", ios::in);
				if (!read1.is_open())
				{
					cerr << "Файл не может быть открыт!\n";
					cerr << -1;
				}
				while (getline(read1, str))
				{
					summ_of_img++;
					vector<Rect> frames, faces;
					Mat image;
					image = imread("C:/Users/Анатолий Долгих/source/repos/CV/Проект4/БДфото/" + str + ".jpg", IMREAD_COLOR);
					//проверка на пустоту
					if (image.empty())
					{
						cerr << "Could not open or find the image" << endl;
						cerr << -1;
					}

					Mat image_gray;
					cvtColor(image, image_gray, COLOR_BGR2GRAY);
					//equalizeHist(frame_gray, frame_gray); // эквилизация

					// забиваем координаты рамок
					unsigned int start_time = clock();
					face_cascade.detectMultiScale(image_gray, frames, step[p], neigh[l], 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
					unsigned int end_time = clock();
					time_ += end_time - start_time;
					summ_of_all_frames += frames.size();

					// забиваем координаты аннотированных лиц
					getline(read1, str);
					int num = atoi(str.c_str());
					summ_of_all_faces += num;
					for (int j = 0; j < num; j++)
					{
						Rect rec;
						int R1, R2, X, Y; //R1 - радиус длины лица(по y), R2 - радиус ширины лица(по х), х,у - коорд. центра прямоуг.

						getline(read1, str, ' ');
						R1 = atoi(str.c_str());

						getline(read1, str, ' ');
						R2 = atoi(str.c_str());

						getline(read1, str, ' ');

						getline(read1, str, ' ');
						X = atoi(str.c_str());

						getline(read1, str, ' ');
						Y = atoi(str.c_str());

						rec.x = X - R2;
						rec.y = Y - R1;
						rec.height = 2 * R1;
						rec.width = 2 * R2;

						faces.push_back(rec);

						getline(read1, str);
					}

					// создаем логическую таблицу ( по горизонтали рамки, по вертикали лица )
					bool** matrix = new bool*[faces.size()];
					for (int j = 0; j < faces.size(); j++) matrix[j] = new bool[frames.size()];
					for (int j = 0; j < faces.size(); j++)
					{
						for (int k = 0; k < frames.size(); k++)
						{
							matrix[j][k] =
								((static_cast<float>(abs(faces[j].x - frames[k].x))) / static_cast<float>(max(faces[j].width, frames[k].width)) <= eps1) &&
								((static_cast<float>(abs(faces[j].y - frames[k].y))) / static_cast<float>(max(faces[j].height, frames[k].height)) <= eps2) &&
								((static_cast<float>(abs(faces[j].x + faces[j].width - frames[k].x - frames[k].width))) / static_cast<float>(max(faces[j].width, frames[k].width)) <= eps3) &&
								((static_cast<float>(abs(faces[j].y + faces[j].height - frames[k].y - frames[k].height))) / static_cast<float>(max(faces[j].height, frames[k].height)) <= eps4);
						}
					}
					// алгоритм анализа
					if (faces.size() == 0)FP += frames.size();
					else
					{
						if (frames.size() == 0)TN += faces.size();
						else
						{
							// FP
							for (int k = 0; k < frames.size(); k++)
							{
								bool F = 0;
								for (int j = 0; j < faces.size(); j++)
								{
									if (matrix[j][k] == 1)
									{
										F = 1;
										break;
									}
								}
								if (F == 0)FP++;
							}
							// TN, TP, DB
							for (int j = 0; j < faces.size(); j++)
							{
								int counter = 0;
								for (int k = 0; k < frames.size(); k++)
								{
									if (matrix[j][k] == 1)
									{
										counter++;
									}
								}
								if (counter == 0)TN++;
								else
								{
									if (counter == 1) TP++;
									else
									{
										TP++;
										DB += counter - 1;
									}
								}
							}
						}
					}
					for (int j = 0; j < faces.size(); j++) delete[] matrix[j];
					delete[] matrix;
				}
				read1.close();
			}
			// запись результатов
			res.open("res_default.csv", ios::app);// | ios::out);
			res << to_string(eps1) << ";" << to_string(step[p]) << ";" << neigh[l] << ";" << TP << ";" << FP << ";" << TN << ";" << DB << ";" << summ_of_img << ";" << summ_of_all_faces << ";" << summ_of_all_frames << ";" << time_ / summ_of_img << endl;
			res.close();
		}
	}
	return 0;
}