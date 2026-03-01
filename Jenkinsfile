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
                withCredentials([file(credentialsId: 'azure-sp', variable: 'AZURE_FILE')]) {
                    bat """
                    az login --service-principal ^
                      --username (Get-Content %AZURE_FILE% | ConvertFrom-Json).clientId ^
                      --password (Get-Content %AZURE_FILE% | ConvertFrom-Json).clientSecret ^
                      --tenant (Get-Content %AZURE_FILE% | ConvertFrom-Json).tenantId
                    """
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
