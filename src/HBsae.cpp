#include <time.h>
#include "generate.h"
#include <gsl/gsl_eigen.h>

extern "C"{
	void BayesFH(double *theta, double *beta, double *Sqsigmav, double *y, double *X, double *b, double *phi, int *n, int *m, int *p, double *beta_prior, double *Sqsigmav_prior, int *betatype, int *Sqsigmavtype)
	{
		// Basic Area Level Fay-Herrior Model

		// if betatype = 0, the prior for beta is normal distributed;otherwise, a flat uniform distribution will be employed

		// if Sqsigmatype = 0, the prior for Sqsigmav would be inv_gamma distribution;otherwise, a uniform distribution will be employed
		
		// Z[i, ] = X[i, ] / b[i]
		gsl_matrix_view gslX = gsl_matrix_view_array(X, *m, *p);
		gsl_vector *gslRec_b = gsl_vector_alloc(*m);
		gsl_vector *gslRec_phi = gsl_vector_alloc(*m);
		double *Recb = gsl_vector_ptr(gslRec_b, 0);
		double *Recphi = gsl_vector_ptr(gslRec_phi, 0);
		for (int i = 0;i < *m;i++){
			Recb[i] = 1.0 / b[i];
			Recphi[i] = 1.0 / phi[i];
		}

		gsl_matrix *gslZ = gsl_matrix_alloc(*m, *p);
		gsl_vector *temp = gsl_vector_alloc(*m);
		for (int i = 0;i < *p;i++){
			gsl_matrix_get_col(temp, &gslX.matrix, i);
			gsl_vector_mul(temp, gslRec_b);
			gsl_matrix_set_col(gslZ, i, temp);
		}
		gsl_vector_view gsly = gsl_vector_view_array(y, *m);

		gsl_vector *gslTheta = gsl_vector_alloc(*m);
		gsl_vector *gslBeta = gsl_vector_alloc(*p);
		double *theta_temp = gsl_vector_ptr(gslTheta, 0);
		double *beta_temp = gsl_vector_ptr(gslBeta, 0);

		// Chole = Z^T Z
		gsl_matrix *Chole = gsl_matrix_calloc(*p, *p);
		gsl_blas_dsyrk(CblasUpper, CblasTrans, 1.0, gslZ, 0.0, Chole);
		for (int i = 1;i < *p;i++)
			for (int j = 0;j < i;j++)
				gsl_matrix_set(Chole, i, j, gsl_matrix_get(Chole, j, i));

		// Sqsigmatype = 0
		if (*Sqsigmavtype == 0){
			// prior specification for theta
			double b0 = *Sqsigmav_prior;

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_phi, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_phi, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
		}

		// Sqsigmatype = 1 
		else{
			// rng generation
			const gsl_rng_type *T1;
			gsl_rng *r1;
			gsl_rng_env_setup();
			T1 = gsl_rng_default;
			r1 = gsl_rng_alloc(T1);
//			unsigned long int s = rand();
			gsl_rng_set(r1, 100 * time(0)); // + s);

			// prior specification for theta
			double eps2 = Sqsigmav_prior[0];

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_phi, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_phi, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}

			// free rng allocation 
			gsl_rng_free(r1);
		}
		
		//free the allocation
		gsl_vector_free(gslRec_b);
		gsl_vector_free(gslRec_phi);
		gsl_vector_free(temp);
		gsl_matrix_free(gslZ);
		gsl_matrix_free(Chole);
		gsl_vector_free(gslTheta);
		gsl_vector_free(gslBeta);
	}

	void BayesUFH(double *theta, double *beta, double *Sqsigmav, double *y, double *X, double *b, double *phi, int *n, int *m, int *p, double *beta_prior, double *Sqsigmav_prior, int *betatype, int *thetatype, int *Sqsigmavtype, int *s)
	{
		// Unmatched Area Level Model

		// if betatype = 0, the prior for beta is normal distributed;otherwise, a flat uniform distribution will be employed

		// if thetatype = 1, dependent variable in the linking model would be log(theta);otherwise logit(theta)

		// if Sqsigmatype = 0, the prior for Sqsigmav would be inv_gamma distribution;otherwise, a uniform distribution will be employed

		// Z[i, ] = X[i, ] / b[i]
		gsl_matrix_view gslX = gsl_matrix_view_array(X, *m, *p);
		gsl_vector *gslRec_b = gsl_vector_alloc(*m);
		gsl_vector *gslRec_phi = gsl_vector_alloc(*m);
		gsl_vector *gslSqrt_phi = gsl_vector_alloc(*m);
		double *Recb = gsl_vector_ptr(gslRec_b, 0);
		double *Recphi = gsl_vector_ptr(gslRec_phi, 0);
		double *Sqrtphi = gsl_vector_ptr(gslSqrt_phi, 0);
		for (int i = 0;i < *m;i++){
			Recb[i] = 1.0 / b[i];
			Recphi[i] = 1.0 / phi[i];
			Sqrtphi[i] = sqrt(phi[i]);
		}

		gsl_matrix *gslZ = gsl_matrix_alloc(*m, *p);
		gsl_vector *temp = gsl_vector_alloc(*m);
		gsl_vector *temp1 = gsl_vector_alloc(*m);
		for (int i = 0;i < *p;i++){
			gsl_matrix_get_col(temp, &gslX.matrix, i);
			gsl_vector_mul(temp, gslRec_b);
			gsl_matrix_set_col(gslZ, i, temp);
		}
		gsl_vector_view gsly = gsl_vector_view_array(y, *m);

		gsl_vector *gslTheta = gsl_vector_alloc(*m);
		gsl_vector *gslBeta = gsl_vector_alloc(*p);
		double *theta_temp = gsl_vector_ptr(gslTheta, 0);
		double *beta_temp = gsl_vector_ptr(gslBeta, 0);

		// Chole = Z^T Z
		gsl_matrix *Chole = gsl_matrix_calloc(*p, *p);
		gsl_blas_dsyrk(CblasUpper, CblasTrans, 1.0, gslZ, 0.0, Chole);
		for (int i = 1;i < *p;i++)
			for (int j = 0;j < i;j++)
				gsl_matrix_set(Chole, i, j, gsl_matrix_get(Chole, j, i));

		// define alpha
		gsl_vector *alpha = gsl_vector_alloc(*m);
		double *alp = gsl_vector_ptr(alpha, 0);

		// rng allocation
		const gsl_rng_type *T;
		gsl_rng *r;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc(T);
//		unsigned long int s1 = rand();
		gsl_rng_set(r, 100 * time(0)); // + s1);

		// define U
		gsl_vector *gslU = gsl_vector_alloc(*m);
		double *U = gsl_vector_ptr(gslU, 0);

		// Sqsigmatype = 0
		if (*Sqsigmavtype == 0){
			// prior specification for theta
			double b0 = *Sqsigmav_prior;

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
		}

		// Sqsigmatype = 1 
		else{
			// rng generation
			const gsl_rng_type *T1;
			gsl_rng *r1;
			gsl_rng_env_setup();
			T1 = gsl_rng_default;
			r1 = gsl_rng_alloc(T1);
//			unsigned long int s2 = rand();
			gsl_rng_set(r1, 100 * time(0)); // + s2);

			// prior specification for theta
			double eps2 = Sqsigmav_prior[0];

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, gslSqrt_phi, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}

			// free rng allocation 
			gsl_rng_free(r1);
		}
		
		//free the allocation
		gsl_vector_free(gslRec_b);
		gsl_vector_free(gslRec_phi);
		gsl_vector_free(gslSqrt_phi);
		gsl_vector_free(alpha);
		gsl_vector_free(temp);
		gsl_vector_free(temp1);
		gsl_matrix_free(gslZ);
		gsl_matrix_free(Chole);
		gsl_vector_free(gslTheta);
		gsl_vector_free(gslBeta);
		gsl_vector_free(gslU);
		gsl_rng_free(r);
	}

	void BayesYC(double *theta, double *beta, double *Sqsigmav, double *Sqsigma, double *y, double *X, double *b, double *phi, int *n, int *m, int *p, double *beta_prior, double *Sqsigmav_prior, double *Sqsigma_prior, int *betatype, int *Sqsigmavtype)
	{
		// Basic Area Level You-Chapman Model

		// if betatype = 0, the prior for beta is normal distributed;otherwise, a flat uniform distribution will be employed

		// if Sqsigmatype = 0, the prior for Sqsigmav would be inv_gamma distribution;otherwise, a uniform distribution will be employed

		// Sqsigmaprior involves (2 * m) parameters, the first m are rate parameters of the inv-gamma distribution, the next m are degree of freedom
		
		// Z[i, ] = X[i, ] / b[i]
		gsl_matrix_view gslX = gsl_matrix_view_array(X, *m, *p);
		gsl_vector *gslRec_b = gsl_vector_alloc(*m);
		gsl_vector *gslRec_phi = gsl_vector_alloc(*m);
		double *Recb = gsl_vector_ptr(gslRec_b, 0);
		double *Recphi = gsl_vector_ptr(gslRec_phi, 0);
		for (int i = 0;i < *m;i++){
			Recb[i] = 1.0 / b[i];
			Recphi[i] = 1.0 / phi[i];
		}

		gsl_matrix *gslZ = gsl_matrix_alloc(*m, *p);
		gsl_vector *temp = gsl_vector_alloc(*m);
		for (int i = 0;i < *p;i++){
			gsl_matrix_get_col(temp, &gslX.matrix, i);
			gsl_vector_mul(temp, gslRec_b);
			gsl_matrix_set_col(gslZ, i, temp);
		}
		gsl_vector_view gsly = gsl_vector_view_array(y, *m);

		gsl_vector *gslTheta = gsl_vector_alloc(*m);
		gsl_vector *gslBeta = gsl_vector_alloc(*p);
		gsl_vector *gslRec_Sqsigma = gsl_vector_alloc(*m);
		double *theta_temp = gsl_vector_ptr(gslTheta, 0);
		double *beta_temp = gsl_vector_ptr(gslBeta, 0);
		double *RecSqsigma_temp = gsl_vector_ptr(gslRec_Sqsigma, 0);

		// Chole = Z^T Z
		gsl_matrix *Chole = gsl_matrix_calloc(*p, *p);
		gsl_blas_dsyrk(CblasUpper, CblasTrans, 1.0, gslZ, 0.0, Chole);
		for (int i = 1;i < *p;i++)
			for (int j = 0;j < i;j++)
				gsl_matrix_set(Chole, i, j, gsl_matrix_get(Chole, j, i));

		// bi[1:m] = Sqsigma_prior[1:m], df[1:m] = Sqsigma_prior[(m+1):end]
		gsl_vector *gslbi = gsl_vector_alloc(*m);
		gsl_vector *gsldf = gsl_vector_alloc(*m);
		double *bi = gsl_vector_ptr(gslbi, 0);
		double *df = gsl_vector_ptr(gsldf, 0);
		for (int i = 0;i < *m;i++){
			bi[i] = Sqsigma_prior[i];
			df[i] = Sqsigma_prior[i+(*m)];
		}

		// Sqsigmatype = 0
		if (*Sqsigmavtype == 0){
			// prior specification for theta
			double b0 = *Sqsigmav_prior;

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_Sqsigma, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_Sqsigma, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
		}

		// Sqsigmatype = 1 
		else{
			// rng generation
			const gsl_rng_type *T1;
			gsl_rng *r1;
			gsl_rng_env_setup();
			T1 = gsl_rng_default;
			r1 = gsl_rng_alloc(T1);
//			unsigned long int s = rand();
			gsl_rng_set(r1, 100 * time(0)); // + s);

			// prior specification for theta
			double eps2 = Sqsigmav_prior[0];

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_Sqsigma, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_vector_memcpy(temp, gslRec_b);
					GenTheta(gslTheta, gslBeta, Sqsigmav, &gsly.vector, &gslX.matrix, gslRec_Sqsigma, temp, i, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}

			// free rng allocation 
			gsl_rng_free(r1);
		}
		
		//free the allocation
		gsl_vector_free(gslRec_b);
		gsl_vector_free(gslRec_phi);
		gsl_vector_free(temp);
		gsl_matrix_free(gslZ);
		gsl_matrix_free(Chole);
		gsl_vector_free(gslTheta);
		gsl_vector_free(gslBeta);
	}

	void BayesUYC(double *theta, double *beta, double *Sqsigmav, double *Sqsigma, double *y, double *X, double *b, double *phi, int *n, int *m, int *p, double *beta_prior, double *Sqsigmav_prior, double *Sqsigma_prior, int *betatype, int *thetatype, int *Sqsigmavtype, int *s)
	{
		// Unmatched Area Level Model

		// if betatype = 0, the prior for beta is normal distributed;otherwise, a flat uniform distribution will be employed

		// if thetatype = 1, dependent variable in the linking model would be log(theta);otherwise logit(theta)

		// if Sqsigmatype = 0, the prior for Sqsigmav would be inv_gamma distribution;otherwise, a uniform distribution will be employed

		// Sqsigmaprior involves (2 * m) parameters, the first m are rate parameters of the inv-gamma distribution, the next m are degree of freedom

		// Z[i, ] = X[i, ] / b[i]
		gsl_matrix_view gslX = gsl_matrix_view_array(X, *m, *p);
		gsl_vector *gslRec_b = gsl_vector_alloc(*m);
		gsl_vector *gslRec_phi = gsl_vector_alloc(*m);
		double *Recb = gsl_vector_ptr(gslRec_b, 0);
		double *Recphi = gsl_vector_ptr(gslRec_phi, 0);
		for (int i = 0;i < *m;i++){
			Recb[i] = 1.0 / b[i];
			Recphi[i] = 1.0 / phi[i];
		}

		gsl_matrix *gslZ = gsl_matrix_alloc(*m, *p);
		gsl_vector *temp = gsl_vector_alloc(*m);
		gsl_vector *temp1 = gsl_vector_alloc(*m);
		gsl_vector *temp2 = gsl_vector_alloc(*m);
		for (int i = 0;i < *p;i++){
			gsl_matrix_get_col(temp, &gslX.matrix, i);
			gsl_vector_mul(temp, gslRec_b);
			gsl_matrix_set_col(gslZ, i, temp);
		}
		gsl_vector_view gsly = gsl_vector_view_array(y, *m);
		double *tem2 = gsl_vector_ptr(temp2, 0);

		gsl_vector *gslTheta = gsl_vector_alloc(*m);
		gsl_vector *gslBeta = gsl_vector_alloc(*p);
		gsl_vector *gslRec_Sqsigma = gsl_vector_alloc(*m);
		double *theta_temp = gsl_vector_ptr(gslTheta, 0);
		double *beta_temp = gsl_vector_ptr(gslBeta, 0);
		double *RecSqsigma_temp = gsl_vector_ptr(gslRec_Sqsigma, 0);

		// Chole = Z^T Z
		gsl_matrix *Chole = gsl_matrix_calloc(*p, *p);
		gsl_blas_dsyrk(CblasUpper, CblasTrans, 1.0, gslZ, 0.0, Chole);
		for (int i = 1;i < *p;i++)
			for (int j = 0;j < i;j++)
				gsl_matrix_set(Chole, i, j, gsl_matrix_get(Chole, j, i));

		// define alpha
		gsl_vector *alpha = gsl_vector_alloc(*m);
		double *alp = gsl_vector_ptr(alpha, 0);

		// rng allocation
		const gsl_rng_type *T;
		gsl_rng *r;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc(T);
//		unsigned long int s1 = rand();
		gsl_rng_set(r, 100 * time(0)); // + s1);

		// define U
		gsl_vector *gslU = gsl_vector_alloc(*m);
		double *U = gsl_vector_ptr(gslU, 0);

		// bi[1:m] = Sqsigma_prior[1:m], df[1:m] = Sqsigma_prior[(m+1):end]
		gsl_vector *gslbi = gsl_vector_alloc(*m);
		gsl_vector *gsldf = gsl_vector_alloc(*m);
		double *bi = gsl_vector_ptr(gslbi, 0);
		double *df = gsl_vector_ptr(gsldf, 0);
		for (int i = 0;i < *m;i++){
			bi[i] = Sqsigma_prior[i];
			df[i] = Sqsigma_prior[i+(*m)];
		}

		// Sqsigmatype = 0
		if (*Sqsigmavtype == 0){
			// prior specification for theta
			double b0 = *Sqsigmav_prior;

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling
				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = exp(theta[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = inv_logit(theta[j]);
				}
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = log(theta_temp[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = logit(theta_temp[j]);
				}

				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / sqrt(RecSqsigma_temp[j]);
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / sqrt(RecSqsigma_temp[j]);
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = exp(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = inv_logit(theta_temp[j]);
					}

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = log(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = logit(theta_temp[j]);
					}

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling
				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = exp(theta[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = inv_logit(theta[j]);
				}
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = log(theta_temp[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = logit(theta_temp[j]);
				}
				

				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, 0, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / sqrt(RecSqsigma_temp[j]);
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / sqrt(RecSqsigma_temp[j]);
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = exp(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = inv_logit(theta_temp[j]);
					}

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = log(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = logit(theta_temp[j]);
					}

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaGam(gslBeta, Sqsigmav, gslZ, temp, i, b0);
				}
			}
		}

		// Sqsigmatype = 1 
		else{
			// rng generation
			const gsl_rng_type *T1;
			gsl_rng *r1;
			gsl_rng_env_setup();
			T1 = gsl_rng_default;
			r1 = gsl_rng_alloc(T1);
//			unsigned long int s2 = rand();
			gsl_rng_set(r1, 100 * time(0)); // + s2);

			// prior specification for theta
			double eps2 = Sqsigmav_prior[0];

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling
				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = exp(theta[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = inv_logit(theta[j]);
				}
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = log(theta_temp[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = logit(theta_temp[j]);
				}
				
				// Generate Sqsigmav[0]
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaNor(gslBeta, &gslBeta0.vector, Sqsigmav, gslZ, Chole, temp, i, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / (sqrt(RecSqsigma_temp[j]));
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / (sqrt(RecSqsigma_temp[j]));
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = exp(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = inv_logit(theta_temp[j]);
					}

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = log(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = logit(theta_temp[j]);
					}

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling
				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = exp(theta[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = inv_logit(theta[j]);
				}
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				if (*thetatype == 1){
					for (int j = 0;j < *m;j++)
						theta_temp[j] = log(theta_temp[j]);
				}
				else{ 
					for (int j = 0;j < *m;j++)
						theta_temp[j] = logit(theta_temp[j]);
				}
				
				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_vector_memcpy(temp, gslTheta);
				gsl_vector_mul(temp, gslRec_b);
				GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, 0, *m, eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenBetaUni(gslBeta, Sqsigmav, gslZ, Chole, temp, i, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					gsl_vector_memcpy(temp1, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					for (int j = 0;j < *m;j++)
						U[j] = gsl_rng_uniform(r);
					if (*thetatype == 1){
						// alpha_i = log(theta_i) + (log(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_add(alpha, gslTheta);
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / sqrt(RecSqsigma_temp[j]);
						GenThetaLog(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					else{
						// alpha_i = log(theta_i) + log(1 - theta_i) + (logit(theta_i) - z_i^T beta)^2 / 2b_i^2 sigma_v^2
						gsl_vector_memcpy(alpha, gslTheta);
						gsl_vector_sub(alpha, temp);
						gsl_vector_mul(alpha, gslRec_b);
						gsl_vector_mul(alpha, alpha);
						gsl_vector_scale(alpha, 0.5 / Sqsigmav[i-1]);
						gsl_vector_sub(alpha, gslTheta);
						for (int k = 0;k < *m;k++)
							alp[k] += 2 * log(inv_logit(theta[(i-1)*(*m)+k]));
						for (int j = 0;j < *m;j++)
							theta_temp[j] = theta[i*(*m)+j];
						for (int j = 0;j < *m;j++)
							tem2[j] = 1.0 / sqrt(RecSqsigma_temp[j]);
						GenThetaLogit(temp1, gslTheta, Sqsigmav, gslRec_b, &gsly.vector, temp2, temp, i, *m, s, U, alpha);
					}
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = exp(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = inv_logit(theta_temp[j]);
					}

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					if (*thetatype == 1){
						for (int j = 0;j < *m;j++)
							theta_temp[j] = log(theta_temp[j]);
					}
					else{ 
						for (int j = 0;j < *m;j++)
							theta_temp[j] = logit(theta_temp[j]);
					}

					// Generate Sqsigmav[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_vector_mul(temp, gslRec_b);
					GenSqsigmaUni(gslBeta, Sqsigmav, gslZ, temp, i, *m, eps2, r1);
				}
			}

			// free rng allocation 
			gsl_rng_free(r1);
		}
		
		//free the allocation
		gsl_vector_free(gslRec_b);
		gsl_vector_free(gslRec_phi);
		gsl_vector_free(alpha);
		gsl_vector_free(temp);
		gsl_vector_free(temp1);
		gsl_vector_free(temp2);
		gsl_matrix_free(gslZ);
		gsl_matrix_free(Chole);
		gsl_vector_free(gslTheta);
		gsl_vector_free(gslBeta);
		gsl_vector_free(gslU);
		gsl_rng_free(r);
	}

	void BayesSFH(double *theta, double *beta, double *Sqsigmav, double *lambda, double *y, double *X, double *phi, double *num, int *li1, int *li2, int *n, int *m, int *p, int *l, double *beta_prior, double *Sqsigmav_prior, int *betatype, int *Sqsigmavtype, int *s)
	{
		// Spatial Area Level Model 

		// if betatype = 0, the prior for beta is normal distributed;otherwise, a flat uniform distribution will be employed

		// if Sqsigmatype = 0, the prior for Sqsigmav would be inv_gamma distribution;otherwise, a uniform distribution will be employed

		gsl_matrix_view gslX = gsl_matrix_view_array(X, *m, *p);
		gsl_vector *gslRec_phi = gsl_vector_alloc(*m);
		double *Recphi = gsl_vector_ptr(gslRec_phi, 0);
		for (int i = 0;i < *m;i++){
			Recphi[i] = 1.0 / phi[i];
		}
		gsl_vector_view gsly = gsl_vector_view_array(y, *m);
		gsl_vector_view diag;

		gsl_vector *temp = gsl_vector_alloc(*m);
		gsl_vector *gslTheta = gsl_vector_alloc(*m);
		gsl_vector *gslBeta = gsl_vector_alloc(*p);
		double *theta_temp = gsl_vector_ptr(gslTheta, 0);
		double *beta_temp = gsl_vector_ptr(gslBeta, 0);

		// Generate neighbouring matrix R
		gsl_matrix *Chole = gsl_matrix_alloc(*m, *m);
		gsl_matrix *gslR = gsl_matrix_calloc(*m, *m);
		double *R = gsl_matrix_ptr(gslR, 0, 0);
		for (int i = 0;i < *m;i++)
			R[i*(*m)+i] = num[i];
		for (int i = 0;i < *l;i++){
			R[li1[i]*(*m)+li2[i]] = -1.0;
			R[li2[i]*(*m)+li1[i]] = -1.0;
		}		
		gsl_matrix *e_vec = gsl_matrix_alloc(*m, *m);
		gsl_vector *e_val = gsl_vector_alloc(*m);
		gsl_eigen_symmv_workspace *works = gsl_eigen_symmv_alloc(*m);
		gsl_eigen_symmv(gslR, e_val, e_vec, works);
		gsl_matrix_set_zero(gslR);
		for (int i = 0;i < *m;i++)
			R[i*(*m)+i] = num[i];
		for (int i = 0;i < *l;i++){
			R[li1[i]*(*m)+li2[i]] = -1.0;
			R[li2[i]*(*m)+li1[i]] = -1.0;
		}

		// rng generation
		const gsl_rng_type *T;
		gsl_rng *r;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc(T);
//		unsigned long int s2 = rand();
		gsl_rng_set(r, 100 * time(0)); // + s2);
		double U;

		// Sqsigmatype = 0
		if (*Sqsigmavtype == 0){
			// prior specification for theta
			double b0 = *Sqsigmav_prior;

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaGam(Sqsigmav, lambda[0], num, temp, li1, li2, 0, *m, *l, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaNor(&gslX.matrix, gslBeta, &gslBeta0.vector, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_phi);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_phi, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaGam(Sqsigmav, lambda[i], num, temp, li1, li2, i, *m, *l, b0);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaGam(Sqsigmav, lambda[0], num, temp, li1, li2, 0, *m, *l, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaUni(&gslX.matrix, gslBeta, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_phi);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_phi, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaGam(Sqsigmav, lambda[i], num, temp, li1, li2, i, *m, *l, b0);
				}
			}
		}

		// Sqsigmatype = 1 
		else{
			// rng generation
			const gsl_rng_type *T1;
			gsl_rng *r1;
			gsl_rng_env_setup();
			T1 = gsl_rng_default;
			r1 = gsl_rng_alloc(T1);
//			unsigned long int s1 = rand();
			gsl_rng_set(r1, 100 * time(0)); // + s1);

			// prior specification for theta
			double eps2 = Sqsigmav_prior[0];

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, 0, *l, *m, lambda[0], eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaNor(&gslX.matrix, gslBeta, &gslBeta0.vector, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_phi);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_phi, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, i, *l, *m, lambda[i], eps2, r1);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling

				// Generate Sqsigmav[0]
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];
				
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, 0, *l, *m, lambda[0], eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaUni(&gslX.matrix, gslBeta, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_phi);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_phi, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, i, *l, *m, lambda[i], eps2, r1);
				}
			}

			// free rng allocation 
			gsl_rng_free(r1);
		}
		
		//free the allocation
		gsl_vector_free(gslRec_phi);
		gsl_vector_free(temp);
		gsl_vector_free(e_val);
		gsl_matrix_free(gslR);
		gsl_matrix_free(e_vec);
		gsl_matrix_free(Chole);
		gsl_vector_free(gslTheta);
		gsl_vector_free(gslBeta);
		gsl_rng_free(r);
		gsl_eigen_symmv_free(works);
	}

	void BayesSYC(double *theta, double *beta, double *Sqsigmav, double *Sqsigma, double *lambda, double *y, double *X, double *phi, double *num, int *li1, int *li2, int *n, int *m, int *p, int *l, double *beta_prior, double *Sqsigmav_prior, double *Sqsigma_prior, int *betatype, int *Sqsigmavtype, int *s)
	{
		// Spatial You-Chapman Model 

		// if betatype = 0, the prior for beta is normal distributed;otherwise, a flat uniform distribution will be employed

		// if Sqsigmatype = 0, the prior for Sqsigmav would be inv_gamma distribution;otherwise, a uniform distribution will be employed

		gsl_matrix_view gslX = gsl_matrix_view_array(X, *m, *p);
		gsl_vector *gslRec_phi = gsl_vector_alloc(*m);
		double *Recphi = gsl_vector_ptr(gslRec_phi, 0);
		for (int i = 0;i < *m;i++){
			Recphi[i] = 1.0 / phi[i];
		}
		gsl_vector_view gsly = gsl_vector_view_array(y, *m);
		gsl_vector_view diag;

		gsl_vector *temp = gsl_vector_alloc(*m);
		gsl_vector *gslTheta = gsl_vector_alloc(*m);
		gsl_vector *gslBeta = gsl_vector_alloc(*p);
		gsl_vector *gslRec_Sqsigma = gsl_vector_alloc(*m);
		double *theta_temp = gsl_vector_ptr(gslTheta, 0);
		double *beta_temp = gsl_vector_ptr(gslBeta, 0);
		double *RecSqsigma_temp = gsl_vector_ptr(gslRec_Sqsigma, 0);

		// Generate neighbouring matrix R
		gsl_matrix *Chole = gsl_matrix_alloc(*m, *m);
		gsl_matrix *gslR = gsl_matrix_calloc(*m, *m);
		double *R = gsl_matrix_ptr(gslR, 0, 0);
		for (int i = 0;i < *m;i++)
			R[i*(*m)+i] = num[i];
		for (int i = 0;i < *l;i++){
			R[li1[i]*(*m)+li2[i]] = -1.0;
			R[li2[i]*(*m)+li1[i]] = -1.0;
		}		
		gsl_matrix *e_vec = gsl_matrix_alloc(*m, *m);
		gsl_vector *e_val = gsl_vector_alloc(*m);
		gsl_eigen_symmv_workspace *works = gsl_eigen_symmv_alloc(*m);
		gsl_eigen_symmv(gslR, e_val, e_vec, works);
		gsl_matrix_set_zero(gslR);
		for (int i = 0;i < *m;i++)
			R[i*(*m)+i] = num[i];
		for (int i = 0;i < *l;i++){
			R[li1[i]*(*m)+li2[i]] = -1.0;
			R[li2[i]*(*m)+li1[i]] = -1.0;
		}

		// bi[1:m] = Sqsigma_prior[1:m], df[1:m] = Sqsigma_prior[(m+1):end]
		gsl_vector *gslbi = gsl_vector_alloc(*m);
		gsl_vector *gsldf = gsl_vector_alloc(*m);
		double *bi = gsl_vector_ptr(gslbi, 0);
		double *df = gsl_vector_ptr(gsldf, 0);
		for (int i = 0;i < *m;i++){
			bi[i] = Sqsigma_prior[i];
			df[i] = Sqsigma_prior[i+(*m)];
		}

		// rng generation
		const gsl_rng_type *T;
		gsl_rng *r;
		gsl_rng_env_setup();
		T = gsl_rng_default;
		r = gsl_rng_alloc(T);
//		unsigned long int s2 = rand();
		gsl_rng_set(r, 100 * time(0)); // + s2);
		double U;

		// Sqsigmatype = 0
		if (*Sqsigmavtype == 0){
			// prior specification for theta
			double b0 = *Sqsigmav_prior;

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaGam(Sqsigmav, lambda[0], num, temp, li1, li2, 0, *m, *l, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaNor(&gslX.matrix, gslBeta, &gslBeta0.vector, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_Sqsigma);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaGam(Sqsigmav, lambda[i], num, temp, li1, li2, i, *m, *l, b0);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaGam(Sqsigmav, lambda[0], num, temp, li1, li2, 0, *m, *l, b0);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaUni(&gslX.matrix, gslBeta, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_Sqsigma);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaGam(Sqsigmav, lambda[i], num, temp, li1, li2, i, *m, *l, b0);
				}
			}
		}

		// Sqsigmatype = 1 
		else{
			// rng generation
			const gsl_rng_type *T1;
			gsl_rng *r1;
			gsl_rng_env_setup();
			T1 = gsl_rng_default;
			r1 = gsl_rng_alloc(T1);
//			unsigned long int s1 = rand();
			gsl_rng_set(r1, 100 * time(0)); // + s1);

			// prior specification for theta
			double eps2 = Sqsigmav_prior[0];

			// betatype = 0
			if(*betatype == 0){
				// prior specification for beta
				gsl_vector_view gslBeta0 = gsl_vector_view_array(beta_prior, *p);
				double eps1 = beta_prior[(*p)]; 

				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, 0, *l, *m, lambda[0], eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaNor(&gslX.matrix, gslBeta, &gslBeta0.vector, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p, eps1);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_Sqsigma);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, i, *l, *m, lambda[i], eps2, r1);
				}
			}
			// betatype = 1
			else{
				// Begin Sampling
				for (int j = 0;j < *m;j++)
					theta_temp[j] = theta[j];
				for (int j = 0;j < *p;j++)
					beta_temp[j] = beta[j];

				// Generate Sqsigma[0]
				for (int j = 0;j < *m;j++)
					RecSqsigma_temp[j] = Sqsigma[j];
				gsl_vector_memcpy(temp, &gsly.vector);
				GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
				for (int j = 0;j < *m;j++)
					Sqsigma[j] = RecSqsigma_temp[j];

				// Generate Sqsigmav[0]
				gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
				gsl_vector_sub(temp, gslTheta);
				GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, 0, *l, *m, lambda[0], eps2, r1);

				for (int i = 1;i < *n;i++){
					// Generate Beta[i]
					for (int j = 0;j < *p;j++)
						beta_temp[j] = beta[i*(*p)+j];
					GenBetaSpaUni(&gslX.matrix, gslBeta, gslTheta, Sqsigmav, num, lambda[i-1], li1, li2, i, *m, *l, *p);
					for (int j = 0;j < *p;j++)
						beta[i*(*p)+j] = beta_temp[j];

					// Generate Theta[i]
					for (int j = 0;j < *m;j++)
						theta_temp[j] = theta[i*(*m)+j];
					gsl_matrix_memcpy(Chole, gslR);
					gsl_matrix_scale(Chole, lambda[i-1]);
					diag = gsl_matrix_diagonal(Chole);
					gsl_vector_add_constant(&diag.vector, 1 - lambda[i-1]);
					gsl_matrix_scale(Chole, 1.0 / Sqsigmav[i-1]);
					gsl_vector_add(&diag.vector, gslRec_Sqsigma);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					GenThetaSpa(gslTheta, &gsly.vector, Chole, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						theta[i*(*m)+j] = theta_temp[j];

					// Generate lambda[i]
					gsl_vector_memcpy(temp, gslTheta);
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, -1.0, temp);
					U = gsl_rng_uniform(r);
					GenLambda(temp, e_val, lambda, Sqsigmav, num, li1, li2, s, i, *m, *l, U);

					// Generate Sqsigma[i]
					for (int j = 0;j < *m;j++)
						RecSqsigma_temp[j] = Sqsigma[i*(*m)+j];
					gsl_vector_memcpy(temp, &gsly.vector);
					GenPhi(gslTheta, gslbi, gsldf, gslRec_phi, gslRec_Sqsigma, temp, *m);
					for (int j = 0;j < *m;j++)
						Sqsigma[i*(*m)+j] = RecSqsigma_temp[j];

					// Generate Sqsigmav[i]
					gsl_blas_dgemv(CblasNoTrans, 1.0, &gslX.matrix, gslBeta, 0.0, temp);
					gsl_vector_sub(temp, gslTheta);
					GenSqsigmaSpaUni(Sqsigmav, temp, num, li1, li2, i, *l, *m, lambda[i], eps2, r1);
				}
			}

			// free rng allocation 
			gsl_rng_free(r1);
		}
		
		//free the allocation
		gsl_vector_free(gslRec_phi);
		gsl_vector_free(gslRec_Sqsigma);
		gsl_vector_free(temp);
		gsl_vector_free(e_val);
		gsl_matrix_free(gslR);
		gsl_matrix_free(e_vec);
		gsl_matrix_free(Chole);
		gsl_vector_free(gslTheta);
		gsl_vector_free(gslBeta);
		gsl_rng_free(r);
		gsl_eigen_symmv_free(works);
	}
}
