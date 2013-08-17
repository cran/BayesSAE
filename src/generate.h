#include<math.h>
#include<R.h>
#include<Rinternals.h>
#include<R_ext/Rdynload.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>

double logit(double x)
{
	return log(x / (1 - x));
}

double inv_logit(double x)
{
	return 1 / (1 + exp(-x));
}

void GenBetaNor(gsl_vector *gslBeta, gsl_vector *gslBeta0, double *Sqsigmav, gsl_matrix *gslZ, gsl_matrix *Chole, gsl_vector *temp, int i, int p, double eps1)
{   
	// temp = theta_i / b_i

	// Chole = Z^T * Z

	// temp1 = eps1 * gslBeta0;
	gsl_vector *temp1 = gsl_vector_alloc(p);
	gsl_vector_memcpy(temp1, gslBeta0);
	gsl_vector_scale(temp1, eps1);

	// tempM = \frac{1}{sigma_v^2} Z^T Z + eps1 
	gsl_matrix *tempM = gsl_matrix_alloc(p, p);
	gsl_matrix_memcpy(tempM, Chole);
	gsl_matrix_scale(tempM, 1.0 / Sqsigmav[i-1]);
	double *tem = gsl_matrix_ptr(tempM, 0, 0);
	for (int j = 0;j < p;j++)
		tem[j+j*p] += eps1;
	gsl_linalg_cholesky_decomp(tempM);

	// beta_star = (Chole) \ (\frac{1}{sigma_v^2} Z^T tilde{theta} + eps1 * beta0)
	gsl_vector *gslbetastar = gsl_vector_alloc(p);
	gsl_blas_dgemv(CblasTrans, 1.0, gslZ, temp, 0.0, gslbetastar);
	gsl_vector_scale(gslbetastar, 1.0 / Sqsigmav[i-1]);
	gsl_vector_add(gslbetastar, temp1);
	gsl_vector_free(temp1);
	gsl_linalg_cholesky_svx(tempM, gslbetastar);

	// beta = gslTheta + upper(Chole) \ rng
	gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, tempM, gslBeta);
	gsl_matrix_free(tempM);
	gsl_vector_add(gslBeta, gslbetastar);
	gsl_vector_free(gslbetastar);
}

void GenBetaSpaNor(gsl_matrix *gslX, gsl_vector *gslBeta, gsl_vector *gslBeta0, gsl_vector *gslTheta, double *Sqsigmav, double *num, double lambda, int *li1, int *li2, int i, int m, int l, int p, double eps1)
{   
	gsl_vector *temp2 = gsl_vector_alloc(p);
	double *theta = gsl_vector_ptr(gslTheta, 0);

	// tempY = K^T X, where K * K^T = R
	gsl_matrix *tempY = gsl_matrix_alloc(l, p);
	gsl_vector_view row1;
	gsl_vector_view row2;
	for (int j = 0;j < l;j++){
		row1 = gsl_matrix_row(tempY, j);
		gsl_matrix_get_row(&row1.vector, gslX, li1[j]);
		row2 = gsl_matrix_row(gslX, li2[j]);
		gsl_vector_sub(&row1.vector, &row2.vector);
	}

	// temp1 = K^T theta
	gsl_vector *temp1 = gsl_vector_alloc(l);
	double *tem1 = gsl_vector_ptr(temp1, 0);
	for (int j = 0;j < l;j++)
		tem1[j] = theta[li1[j]] - theta[li2[j]];

	// temp = \frac{1}{sigma_v^2} X^T (lambda R + (1 - lambda) I) theta + eps1 * gslBeta0
	gsl_vector_memcpy(temp2, gslBeta0);
	gsl_blas_dgemv(CblasTrans, 1-lambda, gslX, gslTheta, eps1 * Sqsigmav[i-1], temp2);
	gsl_blas_dgemv(CblasTrans, lambda, tempY, temp1, 1.0, temp2);
	gsl_vector_free(temp1);
	gsl_vector_scale(temp2, 1.0 / Sqsigmav[i-1]);

	// tempM = \frac{1}{sigma_v^2} X^T (lambda R + (1 - lambda) I) X + eps1
	gsl_matrix *tempM = gsl_matrix_alloc(p, p);
	gsl_matrix_set_identity(tempM);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, (1-lambda)/Sqsigmav[i-1], gslX, gslX, eps1, tempM);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, lambda/Sqsigmav[i-1], tempY, tempY, 1.0, tempM);
	gsl_matrix_free(tempY);
	
	// temp = tempM \ temp + sqrt(tempM) * rng
	gsl_linalg_cholesky_decomp(tempM);
	gsl_linalg_cholesky_svx(tempM, temp2);
	gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, tempM, gslBeta);
	gsl_vector_add(gslBeta, temp2);
	gsl_vector_free(temp2);
	gsl_matrix_free(tempM);
}

//void GenBetaSpaNor1(gsl_matrix *gslX, gsl_vector *gslBeta, gsl_vector *gslBeta0, gsl_vector *gslTheta, double *Sqsigmav, double *num, double lambda, int *li1, int *li2, int i, int m, int l, int p, double eps1)
//{   
//	// tempM =  lambda R + (1 - lambda) I
//	gsl_matrix *tempM = gsl_matrix_calloc(m, m);
//	double *temM = gsl_matrix_ptr(tempM, 0, 0);
//	for (int j = 0;j < l;j++){
//		temM[li1[j]*m+li2[j]] = -1.0;
//		temM[li2[j]*m+li1[j]] = -1.0;
//	}
//	gsl_vector_memcpy(&gsl_matrix_diagonal(tempM).vector, &gsl_vector_view_array(num, m).vector);
//	gsl_matrix_scale(tempM, lambda);
//	gsl_vector_add_constant(&gsl_matrix_diagonal(tempM).vector, 1-lambda);
//
//	// tempY = tempM X
//	gsl_matrix *tempY = gsl_matrix_alloc(m, p);
//	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempM, gslX, 0.0, tempY);
//	gsl_matrix_free(tempM);
//
//	// temp1 = X^T D X / sigma^2 + eps1
//	gsl_matrix *temp1 = gsl_matrix_alloc(p, p);
//	gsl_matrix_set_identity(temp1);
//	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0 / Sqsigmav[i-1], tempY, gslX, eps1, temp1);
//
//	// temp2 = X^T D theta / sigma^2 + eps1 * beta0
//	gsl_vector *temp2 = gsl_vector_alloc(p);
//	gsl_vector_memcpy(temp2, gslBeta0);
//	gsl_blas_dgemv(CblasTrans, 1.0 / Sqsigmav[i-1], tempY, gslTheta, eps1, temp2);
//	gsl_matrix_free(tempY);
//
//	// beta = temp1 \ temp2 + sqrt(temp1) * rng
//	gsl_linalg_cholesky_decomp(temp1);
//	gsl_linalg_cholesky_svx(temp1, temp2);
//	gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, temp1, gslBeta);
//	gsl_vector_add(gslBeta, temp2);
//	gsl_vector_free(temp2);
//	gsl_matrix_free(temp1);
//}

void GenBetaUni(gsl_vector *gslBeta, double *Sqsigmav, gsl_matrix *gslZ, gsl_matrix *Chole, gsl_vector *temp, int i, int p)
{   
	// temp = theta_i / b_i

	// Chole = Z^T Z 
	gsl_matrix *tempM = gsl_matrix_alloc(p, p);
	gsl_matrix_memcpy(tempM, Chole);
	gsl_linalg_cholesky_decomp(tempM);

	// beta_star = (Chole) \ (Z^T temp)
	gsl_vector *gslbetastar = gsl_vector_alloc(p);
	gsl_blas_dgemv(CblasTrans, 1.0, gslZ, temp, 0.0, gslbetastar);
	gsl_linalg_cholesky_svx(tempM, gslbetastar);

	// beta = gslTheta + upper(Chole) \ rng
	gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, tempM, gslBeta);
	gsl_matrix_free(tempM);
	gsl_vector_scale(gslBeta, sqrt(Sqsigmav[i-1]));
	gsl_vector_add(gslBeta, gslbetastar);
	gsl_vector_free(gslbetastar);
}

void GenBetaSpaUni(gsl_matrix *gslX, gsl_vector *gslBeta, gsl_vector *gslTheta, double *Sqsigmav, double *num, double lambda, int *li1, int *li2, int i, int m, int l, int p)
{   
	gsl_vector *temp2 = gsl_vector_alloc(p);
	double *theta = gsl_vector_ptr(gslTheta, 0);

	// tempY = K^T X, where K * K^T = R
	gsl_vector_view row1;
	gsl_vector_view row2;
	gsl_matrix *tempY = gsl_matrix_alloc(l, p);
	for (int j = 0;j < l;j++){
		row1 = gsl_matrix_row(tempY, j);
		gsl_matrix_get_row(&row1.vector, gslX, li1[j]);
		row2 = gsl_matrix_row(gslX, li2[j]);
		gsl_vector_sub(&row1.vector, &row2.vector);
	}

	// temp1 = K^T theta
	gsl_vector *temp1 = gsl_vector_alloc(l);
	double *tem1 = gsl_vector_ptr(temp1, 0);
	for (int j = 0;j < l;j++)
		tem1[j] = theta[li1[j]] - theta[li2[j]];

	// temp = X^T (lambda R + (1 - lambda) I) theta
	gsl_blas_dgemv(CblasTrans, 1-lambda, gslX, gslTheta, 0, temp2);
	gsl_blas_dgemv(CblasTrans, lambda, tempY, temp1, 1.0, temp2);
	gsl_vector_free(temp1);

	// tempM = X^T (lambda R + (1 - lambda) I) X
	gsl_matrix *tempM = gsl_matrix_alloc(p, p);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1-lambda, gslX, gslX, 0, tempM);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, lambda, tempY, tempY, 1.0, tempM);
	gsl_matrix_free(tempY);
	
	// temp = tempM \ temp + sqrt(tempM) * rng * sqrt(Sqsigmav[i-1])
	gsl_linalg_cholesky_decomp(tempM);
	gsl_linalg_cholesky_svx(tempM, temp2);
	gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, tempM, gslBeta);
	gsl_vector_scale(gslBeta, sqrt(Sqsigmav[i-1]));
	gsl_vector_add(gslBeta, temp2);
	gsl_vector_free(temp2);
	gsl_matrix_free(tempM);
}

void GenTheta(gsl_vector *gslTheta, gsl_vector *gslBeta, double *Sqsigmav, gsl_vector *gsly, gsl_matrix *gslX, gsl_vector *gslRec_phi, gsl_vector *temp, int i, int m)
{
	// temp = gslRec_b

	// gamma = (1 ./ phi) ./ (1 ./ phi + 1 ./ (b .^ 2 * Sqsigmav[i-1]))
	gsl_vector_mul(temp, temp);
	gsl_vector_scale(temp, 1.0 / Sqsigmav[i-1]);
	gsl_vector_add(temp, gslRec_phi);
	gsl_vector *gam = gsl_vector_alloc(m);
	gsl_vector_memcpy(gam, gslRec_phi);
	gsl_vector_div(gam, temp);

	// theta = rng * (gamma * phi)
	double *tem = gsl_vector_ptr(temp, 0);
	for (int j = 0;j < m;j++)
		tem[j] = sqrt(tem[j]);
	gsl_vector_div(gslTheta, temp);

	gsl_vector_memcpy(temp, gam);
	gsl_vector_scale(temp, -1.0);
	gsl_vector_add_constant(temp, 1.0);

	// theta = gam .* y + temp .* (X * beta) + theta
	gsl_vector_mul(gam, gsly);
	gsl_vector_add(gslTheta, gam);
	gsl_blas_dgemv(CblasNoTrans, 1.0, gslX, gslBeta, 0.0, gam);
	gsl_vector_mul(gam, temp);
	gsl_vector_add(gslTheta, gam);
	gsl_vector_free(gam);
}

void GenThetaLog(gsl_vector *temp1, gsl_vector *gslTheta, double *Sqsigmav, gsl_vector *gslRec_b, gsl_vector *gsly, gsl_vector *gslSqrt_phi, gsl_vector *temp, int i, int m, int *s, double *U, gsl_vector *alpha)
{
	// temp = gslX * beta

	// theta_i = y_i + sqrt(phi) * rng
	gsl_vector_mul(gslTheta, gslSqrt_phi);
	gsl_vector_add(gslTheta, gsly);

	double *c = gsl_vector_ptr(temp1, 0);
	double *a = gsl_vector_ptr(gslTheta, 0);

	for (int j = 0;j < m;j++){
		if(a[j] <= 0)
			a[j] = c[j];
		else{
			a[j] = log(a[j]);
			s[j] = s[j] - 1;
		}
	}

	// theta_i + (theta_i - z_i^T beta)^2 / (2 b^2 sigmav^2)
	gsl_vector_sub(temp, gslTheta);
	gsl_vector_mul(temp, gslRec_b);
	gsl_vector_mul(temp, temp);
	gsl_vector_scale(temp, 0.5 / Sqsigmav[i-1]);
	gsl_vector_add(temp, gslTheta);

	// acceptance-rejection
	gsl_vector *temp3 = gsl_vector_alloc(m);
	gsl_vector_memcpy(temp3, alpha);
	gsl_vector_sub(temp3, temp);

	double *b = gsl_vector_ptr(temp3, 0);

	for (int j = 0;j < m;j++){
		if(U[j] < exp(b[j]))
			s[j] = s[j] + 1;
		else
			a[j] = c[j];
	}
	gsl_vector_free(temp3);
}

void GenThetaLogit(gsl_vector *temp1, gsl_vector *gslTheta, double *Sqsigmav, gsl_vector *gslRec_b, gsl_vector *gsly, gsl_vector *gslSqrt_phi, gsl_vector *temp, int i, int m, int *s, double *U, gsl_vector *alpha)
{
	// temp = gslZ * beta

	// theta_i = y_i + sqrt(phi) * rng
	gsl_vector_mul(gslTheta, gslSqrt_phi);
	gsl_vector_add(gslTheta, gsly);

	double *a = gsl_vector_ptr(gslTheta, 0);
	double *c = gsl_vector_ptr(temp1, 0);

	for (int j = 0;j < m;j++){
		if(a[j] <= 0 || a[j] >= 1)
			a[j] = c[j];
		else{
			a[j] = logit(a[j]);
			s[j] = s[j] - 1;
		}
	}
				
	// log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / (2 b^2 sigmav^2)
	gsl_vector_sub(temp, gslTheta);
	gsl_vector_mul(temp, gslRec_b);
	gsl_vector_mul(temp, temp);
	gsl_vector_scale(temp, 0.5 / Sqsigmav[i-1]);
	gsl_vector_sub(temp, gslTheta);

	gsl_vector *temp3 = gsl_vector_alloc(m);
	double *b = gsl_vector_ptr(temp3, 0);
	for (int j = 0;j < m;j++)
		b[j] = 2 * log(inv_logit(a[j]));
	gsl_vector_add(temp, temp3);

	// acceptance-rejection
	gsl_vector_memcpy(temp3, alpha);
	gsl_vector_sub(temp3, temp);

	for (int j = 0;j < m;j++){
		if(U[j] < exp(b[j]))
			s[j] = s[j] + 1;
		else
			a[j] = c[j];
	}
	gsl_vector_free(temp3);
}

void GenThetaSpa(gsl_vector *gslTheta, gsl_vector *gsly, gsl_matrix *tempM, gsl_vector *gslRec_phi, gsl_vector *temp, int m)
{
	// temp = X * beta

	// Chole = (E^{-1} + D/Sqsigmav), E = diag(phi), D = \labmbda R + (1 - \lambda) I

	// tempM = Chole
	gsl_linalg_cholesky_decomp(tempM);

	// temp1 = X * beta
	gsl_vector *temp1 = gsl_vector_alloc(m);
	gsl_vector_memcpy(temp1, temp);

	// temp = Chole^{-1} E^{-1} (X beta - y)
	gsl_vector_sub(temp, gsly);
	gsl_vector_mul(temp, gslRec_phi);
	gsl_linalg_cholesky_svx(tempM, temp);

	// temp1 = temp1 - temp
	gsl_vector_sub(temp1, temp);

	// theta = temp1 + Chole^{-0.5} * rng
	gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, tempM, gslTheta);
	gsl_vector_add(gslTheta, temp1);

	gsl_vector_free(temp1);
}

void GenThetaLogSpa(gsl_vector *gslTheta, gsl_vector *temp, gsl_vector *temp1, double *num, double lambda, double* Sqsigmav, double U, int i, int m, int l, int *rate, int *li1, int *li2)
{
	// temp = X * beta

	// temp1 = gslTheta_old
	double *theta_old = gsl_vector_ptr(temp1, 0);
	double *theta_new = gsl_vector_ptr(gslTheta, 0);

	if (gsl_vector_min(gslTheta) <= 0)
		gsl_vector_memcpy(gslTheta, temp1);

	else{
		// beta = sum(log(theta)) + (log(theta) - X beta)^T D * (log(theta) - X beta) / (2 * Sqsigmav)
		double beta = 0;
		for (int j = 0;j < m;j++){
			theta_new[j] = log(theta_new[j]);
			beta += theta_new[j];
		}
		gsl_vector_sub(temp, gslTheta);
		double *tem = gsl_vector_ptr(temp, 0);
		for (int k = 0;k < m;k++)
			beta += pow(tem[k], 2) * (num[k] * lambda + 1 - lambda) / (2.0 * Sqsigmav[i-1]);
		for (int k = 0;k < l;k++)
			beta -= tem[li1[k]] * tem[li2[k]] * lambda / Sqsigmav[i-1];

		// alpha = sum(log(theta')) + (log(theta') - X beta)^T D * (log(theta') - X beta) / (2 * Sqsigmav)
		double alpha = 0;
		for (int k = 0;k < m;k++)
			alpha += theta_old[k];
		gsl_vector_add(temp, gslTheta);
		gsl_vector_sub(temp, temp1);
		for (int k = 0;k < m;k++)
			alpha += pow(tem[k], 2) * (num[k] * lambda + 1 - lambda) / (2.0 * Sqsigmav[i-1]);
		for (int k = 0;k < l;k++)
			alpha -= tem[li1[k]] * tem[li2[k]] * lambda / Sqsigmav[i-1];

		// acceptance-rejection
		double b = alpha - beta;
		if (U < exp(b))
			*rate += 1;
		else
			gsl_vector_memcpy(gslTheta, temp1);
	}
}

void GenThetaLogitSpa(gsl_vector *gslTheta, gsl_vector *temp, gsl_vector *temp1, double *num, double lambda, double* Sqsigmav, double U, int i, int m, int l, int *rate, int *li1, int *li2)
{
	// temp = X * beta

	// temp1 = gslTheta_old
	double *theta_old = gsl_vector_ptr(temp1, 0);
	double *theta_new = gsl_vector_ptr(gslTheta, 0);

	if (gsl_vector_min(gslTheta) <= 0 || gsl_vector_max(gslTheta) >= 1)
		gsl_vector_memcpy(gslTheta, temp1);

	else{
		// beta = sum(log(theta)+log(1-theta)) + (logit(theta) - X beta)^T D * (logit(theta) - X beta) / (2 * Sqsigmav)
		double beta = 0;
		for (int j = 0;j < m;j++){
			beta += (log(theta_new[j]) + log(1 - theta_new[j]));
			theta_new[j] = logit(theta_new[j]);
		}
		gsl_vector_sub(temp, gslTheta);
		double *tem = gsl_vector_ptr(temp, 0);
		for (int k = 0;k < m;k++)
			beta += pow(tem[k], 2) * (num[k] * lambda + 1 - lambda) / (2.0 * Sqsigmav[i-1]);
		for (int k = 0;k < l;k++)
			beta -= tem[li1[k]] * tem[li2[k]] * lambda / Sqsigmav[i-1];

		// alpha = sum(log(theta')+log(1-theta')) + (logit(theta') - X beta)^T D * (logit(theta') - X beta) / (2 * Sqsigmav)
		double alpha = 0;
		for (int k = 0;k < m;k++)
			alpha += (log(inv_logit(theta_old[k])) + log(1 - inv_logit(theta_old[k])));
		gsl_vector_add(temp, gslTheta);
		gsl_vector_sub(temp, temp1);
		for (int k = 0;k < m;k++)
			alpha += pow(tem[k], 2) * (num[k] * lambda + 1 - lambda) / (2.0 * Sqsigmav[i-1]);
		for (int k = 0;k < l;k++)
			alpha -= tem[li1[k]] * tem[li2[k]] * lambda / Sqsigmav[i-1];

		// acceptance-rejection
		double b = alpha - beta;
		if (U < exp(b))
			*rate += 1;
		else
			gsl_vector_memcpy(gslTheta, temp1);
	}
}

void GenSqsigmaUni(gsl_vector *gslBeta, double *Sqsigmav, gsl_matrix *gslZ, gsl_vector *temp, int i, int m, double eps2, gsl_rng *r)
{
	// temp_i = theta_i / b_i

	// lambda = norm(temp - Z beta) ^ 2
	gsl_blas_dgemv(CblasNoTrans, 1.0, gslZ, gslBeta, -1.0, temp);
	double lambda = pow(gsl_blas_dnrm2(temp), 2) / 2.0;

	// generate Sqsigma
	if ((Sqsigmav[i] * lambda) < (1.0 / eps2))
		Sqsigmav[i] = Sqsigmav[i] * lambda;
	else{
		double Sqtemp = lambda / gsl_ran_gamma(r, m / 2.0 - 1.0, 1.0);
		while (Sqtemp >= (1.0 / eps2))
			Sqtemp = lambda / gsl_ran_gamma(r, m / 2.0 - 1.0, 1.0);
		Sqsigmav[i] = Sqtemp;
	}
}

void GenSqsigmaGam(gsl_vector *gslBeta, double *Sqsigmav, gsl_matrix *gslZ, gsl_vector *temp, int i, double b)
{
	// temp_i = theta_i / b_i

	// lambda = norm(tilda{theta} - Z beta) ^ 2
	gsl_blas_dgemv(CblasNoTrans, 1.0, gslZ, gslBeta, -1.0, temp);
	double lambda = pow(gsl_blas_dnrm2(temp), 2) / 2.0;

	// generate Sqsigma
	Sqsigmav[i] = Sqsigmav[i] * (lambda + b);
}

void GenSqsigmaSpaUni(double *Sqsigmav, gsl_vector *temp, double *num, int *li1, int *li2, int i, int l, int m, double lambda, double eps2, gsl_rng *r)
{
	// temp = gslX * gslBeta - theta
	double *tem = gsl_vector_ptr(temp, 0);

	// lambda = temp^T D temp / 2
	double lamb = 0;
	for (int j = 0;j < l;j++)
		lamb -= tem[li1[j]] * tem[li2[j]] * lambda;
	for (int j = 0;j < m;j++)
		lamb += pow(tem[j], 2) * (num[j] * lambda + 1.0 - lambda) / 2.0; 

	// generate Sqsigma
	if ((Sqsigmav[i] * lamb) < (1.0 / eps2))
		Sqsigmav[i] = Sqsigmav[i] * lamb;
	else{
		double Sqtemp = lamb / gsl_ran_gamma(r, m / 2.0 - 1.0, 1.0);
		while (Sqtemp >= (1.0 / eps2))
			Sqtemp = lamb / gsl_ran_gamma(r, m / 2.0 - 1.0, 1.0);
		Sqsigmav[i] = Sqtemp;
	}
}

void GenSqsigmaSpaGam(double *Sqsigmav, double lambda, double *num, gsl_vector *temp, int *li1, int *li2, int i, int m, int l, double b0)
{
	// temp = gslX * gslBeta - theta

	// lambda = temp^T D temp / 2
	double lamb = 0;
	double *tem = gsl_vector_ptr(temp, 0);
	for (int j = 0;j < l;j++)
		lamb -= tem[li1[j]] * tem[li2[j]] * lambda;
	for (int j = 0;j < m;j++)
		lamb += tem[j] * tem[j] * (lambda * num[j] + 1 - lambda) / 2.0;

	// generate Sqsigma
	Sqsigmav[i] = Sqsigmav[i] * (lamb + b0);
}

void GenPhi(gsl_vector *gslTheta, gsl_vector *gslb, gsl_vector *gsldf, gsl_vector *gslRec_phi, gsl_vector *gslphi, gsl_vector *temp, int m)
{
	// Generate Reciprocals of phi

	// temp_i = y_i

	// b_i + ((y_i - theta_i)^2 + d_i s_i^2) / 2
	gsl_vector_sub(temp, gslTheta);
	gsl_vector_mul(temp, temp);
	gsl_vector *temp1 = gsl_vector_alloc(m);
	gsl_vector_memcpy(temp1, gsldf);
	gsl_vector_div(temp1, gslRec_phi);
	gsl_vector_add(temp, temp1);
	gsl_vector_scale(temp, 0.5);
	gsl_vector_add(temp, gslb);
	gsl_vector_free(temp1);

	// generate phi
	gsl_vector_div(gslphi, temp);
}

void GenLambda(gsl_vector *temp, gsl_vector *e_val, double *lambda, double *Sqsigmav, double *num, int *li1, int *li2, int *s,int i, int m, int l, double U)
{
	// temp = X * beta - theta
	
	// alpha = 0.5 log(det(lambda[i-1] R + (1 - lambda[i-1]) I)) - 0.5 / Sqsigmav[i-1] temp^T (lambda[i-1] R + (1 - lambda[i-1]) I) temp
	// beta = 0.5 log(det(lambda[i] R + (1 - lambda[i]) I)) - 0.5 / Sqsigmav[i-1] temp^T (lambda[i] R + (1 - lambda[i]) I) temp
	double alpha = 0;
	double beta = 0;
	double *tem = gsl_vector_ptr(temp, 0);
	double *e_va = gsl_vector_ptr(e_val, 0);
	for (int j = 0;j < m;j++){
		alpha -= pow(tem[j], 2) * (lambda[i-1] * num[j] + 1 - lambda[i-1]) / 2.0;
		beta -= pow(tem[j], 2) * (lambda[i] * num[j] + 1 - lambda[i]) / 2.0;
	}
	for (int j = 0;j < l;j++){
		alpha += tem[li1[j]] * tem[li2[j]] * lambda[i-1];
		beta += tem[li1[j]] * tem[li2[j]] * lambda[i];
	}
	alpha /= Sqsigmav[i-1];
	beta /= Sqsigmav[i-1];
	for (int j = 0;j < m;j++){
		alpha += 0.5 * log(e_va[j] * lambda[i-1] + 1 - lambda[i-1]);
		beta += 0.5 * log(e_va[j] * lambda[i] + 1 - lambda[i]);
	}

	// acceptance rejection
	double b = beta - alpha;
	if (U < exp(b))
		*s = *s + 1;
	else
		lambda[i] = lambda[i-1];
}

