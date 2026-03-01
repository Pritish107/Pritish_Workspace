pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Azure Login') {
            steps {
                withCredentials([file(credentialsId: 'azure-sp', variable: 'AZURE_CREDS')]) {
                    bat '''
                    az login --service-principal --username %CLIENT_ID% --password %CLIENT_SECRET% --tenant %TENANT_ID%
                    '''
                }
            }
        }

        stage('Submit AML Job') {
            steps {
                bat '''
                az extension add -n ml --yes
                az configure --defaults group=pritish_rg workspace=PritishMLWorkspace
                az ml job create --file job.yml
                '''
            }
        }
    }
}
