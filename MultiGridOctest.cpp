/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "Time.h"
#include "MarchingCubes.h"
#include "Octree.h"
#include "SparseMatrix.h"
#include "CmdLineParser.h"
#include "FunctionData.h"
#include "PPolynomial.h"
#include "ply.h"
#include "MemoryUsage.h"

#define SCALE 1.25

#include <stdarg.h>

// the filepath to write comments
char *outputFile = NULL;

// Controls whether to display on the console, 1 means display and 0 means not displayed
int echoStdout = 1;

// Output the comments to the file in the format of "format", and choose whether to output
// the value to the console according to echostdout;
void DumpOutput(const char *format, ...)
{
	if (outputFile)
	{
		FILE *fp = fopen(outputFile, "a");
		va_list args;
		va_start(args, format);
		vfprintf(fp, format, args);
		fclose(fp);
		va_end(args);
	}
	if (echoStdout)
	{
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
	}
}

// Output the comments to the file in the format of "format", and write the comments to str, and choose whether to output
// the value to the console according to echostdout;
void DumpOutput2(char *str, const char *format, ...)
{
	if (outputFile)
	{
		FILE *fp = fopen(outputFile, "a");
		va_list args;
		va_start(args, format);
		vfprintf(fp, format, args);
		fclose(fp);
		va_end(args);
	}

	// output to stdout
	if (echoStdout)
	{
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
	}
	va_list args;
	va_start(args, format);
	vsprintf(str, format, args);
	va_end(args);
	if (str[strlen(str) - 1] == '\n')
	{
		str[strlen(str) - 1] = 0;
	}
}

#include "MultiGridOctreeData.h"

void ShowUsage(char *ex)
{
	printf("Usage: %s\n", ex);
	printf("用法: %s\n", ex); // ex是可执行文件的绝对路径

	printf("\t--in  <input points>\n");
	printf("\t --in  <输入点集>\n");

	printf("\t--out <ouput triangle mesh>\n");
	printf("\t[--out <输出的三角形网格>]\n");

	printf("\t[--depth <maximum reconstruction depth>]\n");
	printf("\t[--depth <最大重建深度>]\n");

	printf("\t\t Running at depth d corresponds to solving on a 2^d x 2^d x 2^d\n");
	printf("\t\t voxel grid.\n");
	printf("\t\t 在深度d运行对应的解为2^d x 2^d x 2^d个体素网格\n");

	printf("\t[--scale <scale factor>]\n");
	printf("\t\t Specifies the factor of the bounding cube that the input\n");
	printf("\t\t samples should fit into.\n");

	printf("\t[--binary]\n");
	printf("\t\t If this flag is enabled, the point set is read in in\n");
	printf("\t\t binary format.\n");

	printf("\t[--solverDivide <subdivision depth>]\n");
	printf("\t[--solverDivide <解算器细分深度>]\n");

	printf("\t\t The depth at which a block Gauss-Seidel solver is used\n");
	printf("\t\t to solve the Laplacian.\n");
	printf("\t\t 用于求解拉普拉斯的阻塞高斯-塞德尔求解器深度\n");

	printf("\t[--samplesPerNode <minimum number of samples per node>\n");
	printf("\t[--samplePerNode <每个节点样本点的最小数目>]\n");
	printf("\t\t This parameter specifies the minimum number of points that\n");
	printf("\t\t should fall within an octree node.\n");
	printf("\t\t 此参数指定了应该落在一个八叉树节点内点的最小数目\n");

	printf("\t[--verbose]\n");
	printf("\t\t 如果这个标志启用, 重构的进度将被输出到STDOUT\n");
}
template <int Degree>
int Execute(int argc, char *argv[])
{
	int i;
	cmdLineString In, Out;
	cmdLineReadable Binary, Verbose, NoResetSamples, NoClipTree;
	cmdLineInt Depth, SolverDivide, IsoDivide, Refine;
	cmdLineInt KernelDepth;
	cmdLineFloat SamplesPerNode;
	cmdLineFloat Scale;
	char *paramNames[] =
		{
			"in", "depth", "out", "refine", "noResetSamples", "noClipTree",
			"binary", "solverDivide", "isoDivide", "scale", "verbose",
			"kernelDepth", "samplesPerNode"};

	// Preset reference command
	cmdLineReadable *params[] =
		{
			&In, &Depth, &Out, &Refine, &NoResetSamples, &NoClipTree,
			&Binary, &SolverDivide, &IsoDivide, &Scale, &Verbose,
			&KernelDepth, &SamplesPerNode};
	int paramNum = sizeof(paramNames) / sizeof(char *);
	int commentNum = 0;
	char **comments;

	// “注释”初始化，有参数可以控制是否输出注释
	comments = new char *[paramNum + 7];
	for (i = 0; i <= paramNum; i++)
	{
		comments[i] = new char[1024];
	}

	const char *Rev = "$Rev: 197 $";
	const char *Date = "$Date: 2006-08-07 10:59:08 -0400 (Mon, 07 Aug 2006) $";

	cmdLineParse(argc - 1, &argv[1], paramNames, paramNum, params, 0);

	//判断是否输出到控制台，Verbose.set为TRUE时输出到控制台
	if (Verbose.set)
	{
		echoStdout = 1;
	}

	// 写入文件是肯定的，是否输出到控制台根据echoStdout是否为1判断，同时写入到comments[commentNum++]
	DumpOutput2(comments[commentNum++], "Running Multi-Grid Octree Surface Reconstructor (degree %d). %s\n", Degree, Rev);

	// 将成功设定过的命令输出到控制台，如果有参数同时输出
	if (In.set)
	{
		DumpOutput2(comments[commentNum++], "\t--in %s\n", In.value);
	}
	if (Out.set)
	{
		DumpOutput2(comments[commentNum++], "\t--out %s\n", Out.value);
	}
	if (Binary.set)
	{
		DumpOutput2(comments[commentNum++], "\t--binary\n");
	}
	if (Depth.set)
	{
		DumpOutput2(comments[commentNum++], "\t--depth %d\n", Depth.value);
	}
	if (SolverDivide.set)
	{
		DumpOutput2(comments[commentNum++], "\t--solverDivide %d\n", SolverDivide.value);
	}
	if (IsoDivide.set)
	{
		DumpOutput2(comments[commentNum++], "\t--isoDivide %d\n", IsoDivide.value);
		printf("\t\t 用于求解拉普拉斯的阻塞高斯-塞德尔求解器深度\n");
	}
	if (Refine.set)
	{
		DumpOutput2(comments[commentNum++], "\t--refine %d\n", Refine.value);
	}
	if (Scale.set)
	{
		DumpOutput2(comments[commentNum++], "\t--scale %f\n", Scale.value);
	}
	if (KernelDepth.set)
	{
		DumpOutput2(comments[commentNum++], "\t--kernelDepth %d\n", KernelDepth.value);
	}
	if (SamplesPerNode.set)
	{
		DumpOutput2(comments[commentNum++], "\t--samplesPerNode %f\n", SamplesPerNode.value);
		printf("\t\t 此参数指定了应该落在一个八叉树节点内点的最小数目\n");
	}
	if (NoResetSamples.set)
	{
		DumpOutput2(comments[commentNum++], "\t--noResetSamples\n");
	}
	if (NoClipTree.set)
	{
		DumpOutput2(comments[commentNum++], "\t--noClipTree\n");
	}

	int solverDivide = 0, isoDivide = 0;
	double t;
	double tt = Time();
	Point3D<float> center;
	Real scale = 1.0;
	Real isoValue = 0;
	Octree<Degree> tree;
	PPolynomial<Degree> ReconstructionFunction = PPolynomial<Degree>::GaussianApproximation();

	center.coords[0] = center.coords[1] = center.coords[2] = 0;
	if (!In.set || !Out.set)
	{
		ShowUsage(argv[0]);
		return 0;
	}

	if (SolverDivide.set)
	{
		solverDivide = SolverDivide.value;
	}
	if (IsoDivide.set)
	{
		isoDivide = IsoDivide.value;
	}

	// 八叉树节点OctNode
	// 给八叉树节点分配空间大小为MEMORY_ALLOCATOR_BLOCK_SIZE
	// 此处只是告知大小，但是没有具体分配空间，只是一个参数的设定而已，没有具体什么动作或操作
	TreeOctNode::SetAllocator(MEMORY_ALLOCATOR_BLOCK_SIZE);
	IsoNodeData::SetAllocator(MEMORY_ALLOCATOR_BLOCK_SIZE);

	t = Time();
	Real scaleFactor = SCALE;
	Real samplesPerNode = 1;
	int kernelDepth, depth = 8, refine = 3;
	if (Depth.set)
	{
		// 如果设定过, 就取设定的值
		depth = Depth.value;
	}

	// 这里好好的为什么要减去2呢？
	kernelDepth = depth - 2;
	if (KernelDepth.set)
	{
		kernelDepth = KernelDepth.value;
	}
	if (SamplesPerNode.set)
	{
		samplesPerNode = SamplesPerNode.value;
	}
	if (Refine.set)
	{
		refine = Refine.value;
	}
	if (Scale.set)
	{
		scaleFactor = Scale.value;
	}

	tree.setFunctionData(ReconstructionFunction, depth, 0, Real(1.0) / (1 << depth));
	DumpOutput("Function Data Set In: %lg\n", Time() - t);
	DumpOutput("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));
	if (kernelDepth > depth)
	{
		// 检测刚才设定的kernelDepth是否合理, 理论上应该小于depth
		fprintf(stderr, "KernelDepth can't be greather than Depth: %d <= %d\n", kernelDepth, depth);
		return EXIT_FAILURE;
	}

	t = Time();
	tree.setTree(In.value,
				 depth,
				 Binary.set,
				 kernelDepth,
				 Real(samplesPerNode),
				 scaleFactor,
				 center,
				 scale,
				 !NoResetSamples.set);
				 
	DumpOutput2(comments[commentNum++], "#             Tree set in: %9.1f (s), %9.1f (MB)\n", Time() - t, tree.maxMemoryUsage);
	DumpOutput("Leaves/Nodes: %d/%d\n", tree.tree.leaves(), tree.tree.nodes());
	DumpOutput("   Tree Size: %.3f MB\n", float(sizeof(TreeOctNode) * tree.tree.nodes()) / (1 << 20));
	DumpOutput("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));

	if (!NoClipTree.set)
	{
		t = Time();
		tree.ClipTree();
		DumpOutput("Tree Clipped In: %lg\n", Time() - t);
		DumpOutput("Leaves/Nodes: %d/%d\n", tree.tree.leaves(), tree.tree.nodes());
		DumpOutput("   Tree Size: %.3f MB\n", float(sizeof(TreeOctNode) * tree.tree.nodes()) / (1 << 20));
	}

	t = Time();
	tree.finalize1(refine);
	DumpOutput("Finalized 1 In: %lg\n", Time() - t);
	DumpOutput("Leaves/Nodes: %d/%d\n", tree.tree.leaves(), tree.tree.nodes());
	DumpOutput("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));

	t = Time();
	tree.maxMemoryUsage = 0;
	// 设置拉普拉斯约束 (这里有OpenMP的并行)
	tree.SetLaplacianWeights();
	DumpOutput2(comments[commentNum++], "#Laplacian Weights Set In: %9.1f (s), %9.1f (MB)\n", Time() - t, tree.maxMemoryUsage);
	DumpOutput("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));

	t = Time();
	tree.finalize2(refine);
	DumpOutput("Finalized 2 In: %lg\n", Time() - t);
	DumpOutput("Leaves/Nodes: %d/%d\n", tree.tree.leaves(), tree.tree.nodes());
	DumpOutput("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));

	tree.maxMemoryUsage = 0;
	t = Time();
	// 设置拉普拉斯矩阵迭代
	// 输入：
	// SolverDivide：解算深度
	// ShowResidual：显示剩余误差
	// MinTters：最小迭代
	// SolverAccuracy：解算精度
	// MaxSolveDepth：最大解算深度
	// FixedIters：修改的迭代，此处值为-1
	tree.LaplacianMatrixIteration(solverDivide);
	DumpOutput2(comments[commentNum++], "# Linear System Solved In: %9.1f (s), %9.1f (MB)\n", Time() - t, tree.maxMemoryUsage);
	// 单位为MB，2的10次方是1k，2的20次方是1M
	DumpOutput("Memory Usage: %.3f MB\n", float(MemoryInfo::Usage()) / (1 << 20));

	CoredVectorMeshData mesh;
	tree.maxMemoryUsage = 0;
	t = Time();
	// 这里有OpenMP并行
	isoValue = tree.GetIsoValue();
	DumpOutput("Got average in: %f\n", Time() - t);
	DumpOutput("Iso-Value: %e\n", isoValue);
	DumpOutput("Memory Usage: %.3f MB\n", float(tree.MemoryUsage()));

	t = Time();
	if (isoDivide)
	{
		tree.GetMCIsoTriangles(isoValue, isoDivide, &mesh);
	}
	else
	{
		tree.GetMCIsoTriangles(isoValue, &mesh);
	}
	DumpOutput2(comments[commentNum++], "#        Got Triangles in: %9.1f (s), %9.1f (MB)\n", Time() - t, tree.maxMemoryUsage);
	DumpOutput2(comments[commentNum++], "#              Total Time: %9.1f (s)\n", Time() - tt);
	PlyWriteTriangles(Out.value, &mesh, PLY_BINARY_NATIVE, center, scale, comments, commentNum);

	return 1;
}

int main(int argc, char *argv[])
{
	int degree = 2;

	switch (degree)
	{
	case 1:
		Execute<1>(argc, argv);
		break;
	case 2:
		Execute<2>(argc, argv);
		break;
	case 3:
		Execute<3>(argc, argv);
		break;
	case 4:
		Execute<4>(argc, argv);
		break;
	case 5:
		Execute<5>(argc, argv);
		break;
	default:
		fprintf(stderr, "Degree %d not supported\n", degree);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
