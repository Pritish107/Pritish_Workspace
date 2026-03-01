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
                    powershell '''
                        $json = Get-Content $env:AZURE_FILE | ConvertFrom-Json
                        $clientId = $json.clientId
                        $clientSecret = $json.clientSecret
                        $tenantId = $json.tenantId
                        $subscriptionId = $json.subscriptionId

                        az login --service-principal `
                          --username $clientId `
                          --password $clientSecret `
                          --tenant $tenantId

                        az account set --subscription $subscriptionId
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
