import spacy
from transformers import pipeline
from typing import Dict, Optional, Tuple, Any, List, Set, DefaultDict
import logging
from datetime import datetime
from pathlib import Path
import re
from dataclasses import dataclass
import pandas as pd
from difflib import SequenceMatcher
import os
from collections import defaultdict

from .document_reader import DocumentReader
from .data_models import ResumeData
from config.settings import settings

logger = logging.getLogger(__name__)

US_VISAS = {
    "h1b": "H-1B Specialty Occupations", "h-1b": "H-1B Specialty Occupations", "l1": "L-1 Intracompany Transfer", "l-1": "L-1 Intracompany Transfer", "f1": "F-1 Student Visa", "f-1": "F-1 Student Visa", "opt": "Optional Practical Training", "cpt": "Curricular Practical Training", "gc": "Green Card", "green card": "Green Card", "us citizen": "US Citizen", "citizen": "US Citizen", "usc": "US Citizen", "ead": "Employment Authorization Document", "tn": "TN NAFTA Professionals", "h4": "H-4 Dependent", "h-4": "H-4 Dependent", "j1": "J-1 Exchange Visitor", "j-1": "J-1 Exchange Visitor", "b1": "B-1 Business Visitor", "b-1": "B-1 Business Visitor", "b2": "B-2 Tourist Visitor", "b-2": "B-2 Tourist Visitor", "o1": "O-1 Extraordinary Ability", "o-1": "O-1 Extraordinary Ability", "e3": "E-3 Specialty Occupation (Australia)", "e-3": "E-3 Specialty Occupation (Australia)", "permanent resident": "Green Card", "lawful permanent resident": "Green Card", "asylee": "Asylee", "refugee": "Refugee"
}
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC"
}
US_STATE_ABBR = {abbr: name.title() for name, abbr in US_STATES.items()}
US_TAX_TERMS = [
    "w2", "w-2", "c2c", "corp to corp", "corp-to-corp", "1099", "contract", "full time", "permanent", "c2h", "contract to hire", "hourly", "salary"
]

@dataclass
class ExtractedValue:
    """Class to hold extracted values with confidence scores and metadata."""
    
    def __init__(self, value: Any, confidence: float, method: str, structured_data: Optional[Dict] = None):
        self.value = value
        self.confidence = confidence
        self.method = method
        self.structured_data = structured_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'value': self.value,
            'confidence': self.confidence,
            'method': self.method,
            'structured_data': self.structured_data
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value} (confidence: {self.confidence:.2f}, method: {self.method})"

class ResumeParser:
    """Resume parser with improved extraction methods"""
    
    # Common skills categories
    COMMON_SKILLS = {
        # Technical Skills
        "programming": [
            "python", "java", "javascript", "typescript", "c#", "c++", "ruby", "php", "swift", "kotlin",
            "go", "rust", "scala", "perl", "r", "matlab", "sql", "pl/sql", "t-sql", "nosql",
            "html", "css", "sass", "less", "xml", "json", "yaml", "markdown", "bash", "powershell",
            "assembly", "cobol", "fortran", "pascal", "lisp", "prolog", "haskell", "erlang", "elixir",
            "dart", "objective-c", "visual basic", "groovy", "clojure", "f#", "ocaml", "julia",
            "vba", "autohotkey", "zig", "nim", "crystal", "d", "hack",
            "idl", "gams", "sas", "stata", "spss", "apl", "j", "k",
            "verilog", "vhdl", "systemverilog", "ada", "mips", "arm", "risc-v", "x86",
            "agda", "coq", "idris", "elm", "purescript", "mercury", "janet", "rego",
            "toml", "ini", "hcl", "makefile", "cmake", "bazel", "dockerfile", "compose yaml",
            "solidity", "vyper", "move", "cadence",
            "brainfuck", "whitespace", "befunge", "logo", "scratch",
            "smalltalk", "modula-2", "modula-3", "oberon", "eiffel", "rexx", "tcl", "snobol", "icon", "unicon",
            "tcsh", "csh", "zsh", "ksh", "fish", "expect", "gnuplot",
            "mathcad", "maxima", "maple", "mupad", "labview",
            "ant", "gradle", "nix", "puppet", "chef", "saltstack",
            "biopython", "bioperl", "ampl", "sbml", "cellml",
            "ink!", "plutus", "marlowe", "scilla",
            "tla+", "alloy", "red", "nial",
            "graphql", "smt-lib", "z3", "scss", "xslt",
            "gdscript", "blueprints", "qml", "haxe",
            "malbolge", "piet", "shakespeare", "chef", "cow", "ook!"
        ],
        "frameworks": [
            "react", "angular", "vue", "node.js", "express", "django", "flask", "spring", "laravel",
            "ruby on rails", "asp.net", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
            "numpy", "jquery", "bootstrap", "tailwind", "material-ui", "redux", "graphql", "rest",
            "soap", "grpc", "websocket", "socket.io", "next.js", "nuxt.js", "gatsby", "d3.js",
            "ember.js", "backbone.js", "meteor", "svelte", "alpine.js", "stimulus", "lit", "preact",
            "fastapi", "fastify", "koa", "hapi", "nest.js", "adonis.js", "loopback", "strapi",
            "symfony", "codeigniter", "cakephp", "phalcon", "yii", "zend", "fuelphp", "slim",
            "play", "akka", "micronaut", "quarkus", "vert.x", "ktor", "jhipster", "grails",
            "gin", "echo", "fiber", "beego", "revel", "buffalo", "iris", "chi",
            "rocket", "actix", "axum", "tide", "warp", "hyper", "tokio", "async-std",
            "blazor", "razor", "htmx", "remix", "solid.js", "marko", "inferno", "qwik", "astro",
            "electron", "tauri", "capacitor", "cordova", "ionic", "nativeScript", "expo", "react native",
            "flutter", "jetpack compose", "swiftUI", "xamarin", "unity", "unreal engine", "godot",
            "mlpack", "xgboost", "lightgbm", "catboost", "huggingface transformers", "openCV",
            "openvino", "onnx runtime", "mlflow", "ray", "dask", "modin",
            "airflow", "luigi", "prefect", "dagster", "kedro",
            "kafka streams", "spark streaming", "flink", "nifi", "beam",
            "serverless", "vercel", "netlify", "firebase", "supabase", "amplify",
            "terraform", "pulumi", "ansible", "chef", "puppet", "saltstack",
            "jest", "mocha", "chai", "cypress", "playwright", "vitest", "jasmine",
            "selenium", "robot framework", "testng", "junit", "nunit", "xunit",
            "openresty", "kong", "traefik", "envoy", "caddy",
            "ember fastboot", "spring boot", "spring security", "spring cloud",
            "dotnet core", "dotnet mvc", "express.js", "sanic", "falcon",
            "bottle", "tornado", "web2py", "pyramid", "hug",
            "avalonia", "gtk", "qt", "wxwidgets"
        ],
        "databases": [
            "mysql", "postgresql", "oracle", "sql server", "mongodb", "cassandra", "redis",
            "elasticsearch", "dynamodb", "couchbase", "neo4j", "firebase", "cosmos db",
            "mariadb", "sqlite", "hbase", "influxdb", "couchdb", "arangodb", "rethinkdb",
            "db2", "sybase", "teradata", "vertica", "greenplum", "snowflake", "bigquery",
            "redshift", "aurora", "documentdb", "timestream", "keyspaces", "opensearch",
            "meilisearch", "typesense", "algolia", "solr", "sphinx", "manticore",
            "clickhouse", "timescaledb", "questdb", "prometheus", "victoria metrics",
            "trino", "presto", "druid", "pinot", "apache iceberg", "delta lake", "hudi",
            "duckdb", "sqlite3", "firebird", "interbase", "informix",
            "hazelcast", "etcd", "leveldb", "rocksdb", "badgerdb", "tokudb",
            "yugabyte", "cockroachdb", "tidb", "tarantool", "memcached",
            "janusgraph", "orientdb", "gremlin", "tigergraph",
            "milvus", "weaviate", "pinecone", "qdrant", "vespa",
            "zincsearch", "tantivy", "xapian", "whoosh",
            "simpledb", "cloud firestore", "faiss", "annoy", "nmslib"
        ],
        "cloud": [
            "aws", "azure", "gcp", "heroku", "digitalocean", "linode", "vultr", "cloudflare",
            "s3", "ec2", "lambda", "rds", "dynamodb", "cloudfront", "route53", "vpc",
            "iam", "sagemaker", "rekognition", "comprehend", "transcribe", "translate",
            "app engine", "cloud functions", "cloud run", "cloud sql", "bigquery",
            "compute engine", "cloud storage", "cloud pub/sub", "cloud spanner",
            "app service", "functions", "cosmos db", "blob storage", "cdn", "traffic manager",
            "virtual machines", "kubernetes", "docker", "terraform", "ansible", "jenkins",
            "github actions", "gitlab ci", "circleci", "travis ci", "aws compliance",
            "aws support", "azure devops", "cloud security", "cloud architecture",
            "alibaba cloud", "oracle cloud", "ibm cloud", "rackspace", "ovh", "scaleway",
            "cloudflare workers", "vercel", "netlify", "render", "fly.io", "railway",
            "cloudflare pages", "cloudflare r2", "cloudflare d1", "cloudflare kv",
            "cloudflare durable objects", "cloudflare workers kv", "cloudflare workers sites",
            "cloudflare workers unbound", "cloudflare workers durables", "cloudflare workers cron",
            "cloudflare workers queue", "cloudflare workers streams", "cloudflare workers websockets",
            "lightsail", "batch", "step functions", "app mesh", "fargate", "eks", "amplify", "codepipeline",
            "codebuild", "codecommit", "codeartifact", "cloudformation", "control tower", "cloudwatch",
            "x-ray", "guardduty", "inspector", "macie", "waf", "shield", "elastic beanstalk", "athena",
            "aws glue", "aws lake formation", "aws data pipeline", "outposts", "local zones", "snowball",
            "azure blob", "azure file storage", "azure container instances", "aks", "azure monitor",
            "azure sentinel", "azure advisor", "azure bastion", "azure firewall", "azure key vault",
            "azure logic apps", "azure event grid", "azure event hub", "azure api management",
            "gke", "gcs", "vertex ai", "autoML", "dataproc", "dataflow", "composer", "cloud dataprep",
            "cloud armor", "secret manager", "workflows", "iap", "beyondcorp", "operations suite",
            "artifact registry", "cloud deploy", "run jobs", "google cloud marketplace",
            "cloud dns", "cloud nat", "interconnect", "cloud router", "filestore",
            "wasabi", "backblaze b2", "upcloud", "phoenixNAP", "cyon", "hetzner", "exoscale",
            "ovhcloud", "ionos cloud", "stackpath", "digitalocean app platform", "scaleway serverless",
            "akamai connected cloud", "minio", "openstack", "harvester", "longhorn", "k3s", "rancher"
        ],
        "devops": [
            "docker", "kubernetes", "jenkins", "gitlab ci", "github actions", "circleci",
            "travis ci", "ansible", "terraform", "puppet", "chef", "prometheus", "grafana",
            "elk stack", "splunk", "datadog", "new relic", "nagios", "zabbix", "consul",
            "vault", "istio", "linkerd", "helm", "argo", "spinnaker", "drone", "bamboo",
            "teamcity", "git", "svn", "mercurial", "bitbucket", "github", "gitlab",
            "rancher", "openshift", "mesos", "marathon", "nomad", "etcd",
            "fluentd", "logstash", "filebeat", "metricbeat", "packetbeat", "heartbeat",
            "auditbeat", "journalbeat", "functionbeat", "winlogbeat", "cloudwatch",
            "cloudtrail", "cloudfront", "route53", "vpc", "iam", "sagemaker", "rekognition",
            "comprehend", "transcribe", "translate", "app engine", "cloud functions",
            "cloud run", "cloud sql", "bigquery", "compute engine", "cloud storage",
            "cloud pub/sub", "cloud spanner", "app service", "functions", "cosmos db",
            "blob storage", "cdn", "traffic manager", "virtual machines", "aws compliance",
            "aws support", "azure devops", "cloud security", "cloud architecture",
            "codepipeline", "codebuild", "codecommit", "codeartifact", "azure pipelines",
            "aws codedeploy", "tekton", "flux", "tilt", "skaffold", "werf", "goreleaser",
            "nexus", "jfrog artifactory", "sonatype nexus", "harbor", "quay", "chartmuseum",
            "packer", "vagrant", "cloud-init", "kustomize", "argocd", "chaos mesh",
            "gremlin", "litmus", "thanos", "victoria metrics", "loki", "tempo",
            "otel collector", "opentelemetry", "jaeger", "zipkin", "sentry",
            "rollbar", "raygun", "appdynamics", "dynatrace", "statuspage", "pagerduty",
            "uptime kuma", "netdata", "bosun", "cabourot", "glances", "cAdvisor",
            "sysdig", "falco", "osquery", "clamav", "clamwin", "tripwire", "auditd",
            "docker compose", "podman", "buildah", "containerd", "crio", "minikube", "kind",
            "k3s", "longhorn", "velero", "restic", "stern", "kubectl", "k9s",
            "argus", "azure monitor", "azure sentinel", "aws x-ray", "aws guardduty",
            "aws inspector", "aws macie", "aws config", "aws organizations", "aws control tower",
            "cloudformation", "cdk", "bicep", "pulumi", "sops", "secrets manager",
            "aws parameter store", "keycloak", "auth0", "okta", "vault", "doppler"
        ],
        "methodologies": [
            "agile", "scrum", "kanban", "waterfall", "devops", "ci/cd", "tdd", "bdd",
            "extreme programming", "lean", "six sigma", "itil", "cmmi", "pmi", "pmp",
            "prince2", "safe", "crystal", "fdd", "dsdm", "rad", "rup", "v-model",
            "spiral", "prototype", "incremental", "iterative", "lean six sigma",
            "rapid application development", "joint application development", "feature driven development",
            "dynamic systems development method", "rational unified process", "unified process",
            "disciplined agile delivery", "scaled agile framework", "large-scale scrum",
            "scrum of scrums", "nexus", "scrum at scale", "scrum@scale", "scrum of scrums",
            "scrum of scrums of scrums", "scrum of scrums of scrums of scrums", "scrum of scrums of scrums of scrums of scrums",
            "xp", "agile modeling", "agile unified process", "lean software development",
            "agile project management", "agile release train", "value stream mapping",
            "mob programming", "pair programming", "shift-left testing", "site reliability engineering",
            "chaos engineering", "continuous integration", "continuous delivery",
            "continuous deployment", "continuous testing", "infrastructure as code",
            "bizdevops", "secdevops", "devsecops", "test-driven infrastructure",
            "agile portfolio management", "systems development life cycle", "msdlc",
            "dual-track agile", "event storming", "impact mapping", "story mapping",
            "specification by example", "design thinking", "design sprint", "agile architecture",
            "lean startup", "product-led growth", "outcome-driven development",
            "object-oriented analysis and design", "domain-driven design",
            "service-oriented architecture", "microservices architecture", "monolithic architecture",
            "platform engineering", "inner source", "trunk-based development", "gitflow",
            "model-driven engineering", "behavior-driven infrastructure", "safe@scale"
        ],
        "soft_skills": [
            # Communication Skills
            "verbal communication", "written communication", "public speaking", "presentation",
            "active listening", "nonverbal communication", "body language", "clarity", "storytelling",
            "interpersonal communication", "cross-functional communication",

            # Collaboration & Teamwork
            "teamwork", "collaboration", "relationship building", "team building", "team leadership",
            "team management", "team motivation", "team development", "team coaching", "team mentoring",
            "team facilitation", "team consulting", "team training", "team innovation", "team problem solving",

            # Leadership & Management
            "leadership", "delegation", "decision making", "strategic thinking", "vision setting",
            "influence", "motivation", "mentoring", "coaching", "initiative", "accountability",
            "reliability", "ethics", "change management", "stakeholder management", "performance management",

            # Problem Solving & Thinking Skills
            "problem solving", "critical thinking", "analytical thinking", "creative thinking",
            "innovation", "research", "analysis", "strategic thinking", "attention to detail",
            "root cause analysis", "solution design", "logical reasoning",

            # Organization & Productivity
            "time management", "prioritization", "organization", "planning", "goal setting",
            "productivity", "work ethic", "self-management", "multitasking",

            # Emotional & Social Intelligence
            "emotional intelligence", "empathy", "self-awareness", "self-regulation", "social skills",
            "interpersonal skills", "patience", "diplomacy", "tact", "respectfulness", "trust-building",
            "conflict resolution", "mediation", "relationship management",

            # Flexibility & Adaptability
            "adaptability", "resilience", "flexibility", "stress management", "open-mindedness",
            "learning agility", "ability to work under pressure", "grit",

            # Influencing & Negotiation
            "negotiation", "persuasion", "influence", "networking", "diplomacy", "tactful communication",

            # Training & Facilitation
            "facilitation", "training", "instruction", "knowledge sharing", "workshop delivery",
            "onboarding", "upskilling others", "continuous improvement culture",

            # Consulting & Client Management
            "consulting", "client interaction", "requirements gathering", "expectation management",
            "business relationship management", "solution presentation",

            # Documentation & Reporting
            "documentation", "report writing", "business writing", "technical writing", "meeting notes",
            "process documentation", "project reporting"
        ],
        "business_skills": [
            "business analysis", "requirements gathering", "requirements analysis", "gap analysis",
            "impact analysis", "feasibility study", "cost-benefit analysis", "SWOT analysis",
            "KPI analysis", "data analysis", "root cause analysis", "trend analysis",
            "risk analysis", "compliance analysis", "competitive analysis", "impact assessment",
            "functional requirements", "non-functional requirements", "business requirements",
            "user stories", "use cases", "acceptance criteria", "requirement traceability",
            "BRD", "FRD", "SRS", "process documentation", "SOP development", "workflow documentation",
            "strategic planning", "business planning", "roadmap planning", "vision alignment",
            "operational planning", "organizational analysis", "organizational design",
            "business model design", "go-to-market strategy", "business case development",
            "process mapping", "process modeling", "process improvement", "process reengineering",
            "process standardization", "process automation", "process integration",
            "process optimization", "lean process management", "six sigma", "BPMN modeling",
            "value stream mapping", "workflow optimization",
            "stakeholder analysis", "stakeholder engagement", "client communication",
            "interviewing", "facilitation", "workshop leadership", "elicitation techniques",
            "active listening", "presentation skills", "conflict resolution", "negotiation",
            "change communication", "meeting facilitation",
            "agile methodology", "scrum ceremonies", "product backlog grooming", "story mapping",
            "MVP definition", "prioritization", "agile estimation", "release planning",
            "sprint planning", "retrospectives", "JIRA usage", "confluence documentation",
            "change management", "organizational change", "digital transformation",
            "business transformation", "agile transformation", "innovation management",
            "continuous improvement", "readiness assessment", "adoption planning",
            "business intelligence", "dashboarding", "data visualization", "reporting",
            "tableau", "power bi", "excel modeling", "erp analysis", "crm analysis",
            "governance frameworks", "regulatory compliance", "audit support",
            "policy development", "SLA definition", "data privacy compliance",
            "GDPR awareness", "SOX compliance", "user acceptance testing", "test case definition", "test scenario mapping",
            "UAT coordination", "quality assurance processes", "defect tracking",
            "cross-functional collaboration", "team leadership", "problem solving",
            "analytical thinking", "critical thinking", "decision making", "adaptability",
            "time management", "prioritization", "emotional intelligence"
        ],
        "data_skills": [
            #Core Concepts
            "data analysis", "data analytics", "data management", "data governance", "data quality",
            "data privacy", "data security", "data protection", "data compliance", "data auditing",
            "data integrity", "data lineage", "data cataloging", "data classification",

            #Engineering & Architecture
            "data engineering", "data pipelines", "etl", "elt", "data lakes",
            "data warehouses", "data marts", "data mesh", "data fabric", "data architecture",
            "real-time data", "batch processing", "stream processing", "data modeling",
            "dimensional modeling", "star schema", "snowflake schema",

            #Science & ML
            "data science", "machine learning", "statistical modeling", "predictive modeling",
            "model evaluation", "model deployment", "feature engineering", "algorithm selection",
            "data experimentation", "ab testing", "causal inference",

            #Visualization & Reporting
            "data visualization", "data storytelling", "dashboard creation", "interactive dashboards",
            "reporting", "kpi tracking", "metrics development", "business intelligence",

            #Tools(must-have for resume matching)
            "excel", "tableau", "power bi", "lookml", "looker", "superset", "metabase",
            "qlikview", "qliksense", "google data studio", "mode analytics", "plotly", "dash",
            "matplotlib", "seaborn", "ggplot", "d3.js", "highcharts", "chart.js",

            #Programming & Query Languages
            "sql", "pl/sql", "t-sql", "nosql", "mongodb", "spark sql",
            "python", "r", "scala", "pyspark", "bash", "java", "sas",

            #Platforms & Frameworks
            "spark", "hadoop", "hive", "pig", "airflow", "dbt", "kafka", "flink",
            "aws glue", "aws athena", "google bigquery", "azure synapse", "databricks",
            "snowflake", "redshift", "presto", "trino", "delta lake", "iceberg", "hudi",

            #Standards & Quality
            "data standardization", "data normalization", "data validation", "data reconciliation",
            "data profiling", "data cleansing", "data wrangling", "data harmonization",
            "data enrichment", "data transformation", "data deduplication",

            #Integration & APIs
            "data integration", "api integration", "restful apis", "graphql apis",
            "data ingestion", "cdc", "webhooks", "data sync", "data connectors",

            #Collaboration & Strategy
            "data strategy", "data roadmap", "data stewardship", "data literacy",
            "data democratization", "data enablement", "data operations", "data observability",

            #Governance Tools
            "collibra", "informatica", "alation", "talend", "azure purview", "google datacatalog",
            "aws glue catalog", "octopai", "dataedo", "erwin",

            #Cloud-specific Data Services
            "aws redshift", "aws glue", "aws athena", "s3 data lake", "azure data factory",
            "azure synapse", "azure databricks", "gcp bigquery", "gcp dataproc", "gcp composer"
        ],
        "domain_specific": [
            #Core Industries
            "healthcare", "medical", "clinical", "pharmaceutical", "biotechnology", "life sciences",
            "finance", "banking", "investment banking", "capital markets", "insurance", "fintech",
            "retail", "ecommerce", "consumer goods", "wholesale", "fashion", "luxury",
            "manufacturing", "automotive", "industrial", "aerospace", "defense", "semiconductors",
            "logistics", "transportation", "shipping", "fleet management", "supply chain", "warehousing",
            "energy", "utilities", "oil and gas", "renewables", "power generation", "nuclear",
            "telecommunications", "media", "broadcast", "publishing", "streaming", "advertising",
            "entertainment", "gaming", "animation", "film production", "music", "sports",
            "education", "edtech", "higher education", "k-12", "online learning", "corporate training",
            "government", "public sector", "military", "space", "municipal services", "policy",
            "non-profit", "ngo", "social impact", "public health", "charity", "development aid",
            "construction", "architecture", "urban planning", "infrastructure", "real estate",
            "engineering", "civil engineering", "mechanical engineering", "electrical engineering",
            "design", "ux design", "product design", "graphic design", "industrial design",

            #Business Functions
            "marketing", "digital marketing", "content marketing", "seo", "sem", "ppc", "branding",
            "sales", "inside sales", "b2b sales", "b2c sales", "pre-sales", "post-sales", "crm",
            "customer service", "customer experience", "technical support", "client relations",
            "human resources", "recruiting", "talent acquisition", "employee relations", "hr tech",
            "legal", "compliance", "regulatory affairs", "intellectual property", "contracts",

            #Risk, Audit, Security
            "risk management", "audit", "fraud detection", "compliance monitoring", "governance",
            "security", "cybersecurity", "information security", "application security",
            "network security", "cloud security", "data security", "endpoint security",
            "mobile security", "web security", "api security", "database security", "server security",
            "infrastructure security", "operational security", "identity and access management",
            "penetration testing", "threat intelligence", "incident response", "security operations center",

            #Tech-specific Business Domains
            "cloud computing", "big data", "artificial intelligence", "machine learning",
            "internet of things", "blockchain", "robotics", "edge computing", "quantum computing",
            "augmented reality", "virtual reality", "digital twins", "5g", "wearables",
            
            #Other/Niche Domains
            "hospitality", "travel", "tourism", "food and beverage", "agriculture", "farming",
            "mining", "marine", "aviation", "railway", "logistics tech", "healthtech", "proptech",
            "agritech", "greentech", "legaltech", "insurtech", "martech", "regtech", "edtech", "climatetech"
        ],
        "sdlc": [
            # Requirements Phase
            "requirements elicitation", "requirements gathering", "requirements workshops",
            "interviews", "focus groups", "stakeholder interviews", "joint application design",
            "brainstorming", "storyboarding", "surveys", "questionnaires", "observation",
            "business requirements", "system requirements", "software requirements",
            "functional requirements", "non-functional requirements", "technical requirements",
            "regulatory requirements", "security requirements", "compliance requirements",
            "user stories", "epics", "use cases", "user personas", "stakeholder analysis",
            "requirement traceability matrix", "requirements documentation",
            "requirements prioritization", "requirements modeling", "requirements review",
            "requirements validation", "requirements sign-off", "scope definition",
            "acceptance criteria", "moscow prioritization", "kano analysis",
            "volere specification", "requirements change control",

            # Design Phase
            "system design", "software design", "application design", "solution architecture",
            "software architecture", "microservices architecture", "event-driven architecture",
            "cloud architecture", "hybrid cloud design", "serverless architecture",
            "security architecture", "data architecture", "network architecture",
            "enterprise architecture", "technical architecture", "api design",
            "interface design", "ui/ux design", "wireframes", "mockups", "prototypes",
            "design patterns", "object-oriented design", "component design", "modular design",
            "uml diagrams", "sequence diagrams", "class diagrams", "activity diagrams",
            "er diagrams", "data flow diagrams", "low-level design", "high-level design",
            "architecture diagrams", "service blueprint", "design documentation",
            "design validation", "design review", "style guide", "branding guideline",

            # Development Phase
            "software development", "code implementation", "feature development",
            "bug fixing", "code optimization", "refactoring", "peer programming",
            "pair programming", "agile development", "scrum development",
            "version control", "git branching", "git flow", "commit management",
            "merge conflict resolution", "repository management", "monorepo strategies",
            "code documentation", "code commenting", "continuous integration",
            "continuous delivery", "continuous deployment", "ci/cd pipeline",
            "build automation", "script development", "api development",
            "microservices development", "containerized development",
            "dependency management", "package management", "code reviews",
            "linting", "static code analysis", "secure coding practices",
            "test-driven development", "behavior-driven development",
            "unit testing", "mock testing", "development sprints", "sprint planning",

            # Testing Phase
            "test planning", "test strategy", "test estimation", "test scheduling",
            "test case development", "test scenario creation", "test execution",
            "unit testing", "integration testing", "system testing", "acceptance testing",
            "user acceptance testing", "regression testing", "smoke testing",
            "sanity testing", "performance testing", "load testing", "stress testing",
            "security testing", "penetration testing", "fuzz testing", "vulnerability assessment",
            "api testing", "ui testing", "cross-browser testing", "compatibility testing",
            "mobile testing", "accessibility testing", "test automation",
            "selenium", "junit", "testng", "pytest", "robot framework",
            "k6", "postman", "soapui", "bug tracking", "test reporting",
            "test metrics", "defect management", "test coverage analysis",
            "traceability matrix", "test closure",

            # Deployment Phase
            "release planning", "release notes", "deployment strategy",
            "blue-green deployment", "canary deployment", "rolling deployment",
            "zero downtime deployment", "deployment automation", "infrastructure as code",
            "configuration management", "environment configuration", "deployment scripts",
            "deployment pipelines", "ci/cd tools", "deployment to cloud", "container deployment",
            "kubernetes deployment", "deployment verification", "deployment testing",
            "release coordination", "rollback strategy", "hotfix deployment",
            "emergency releases", "pre-prod deployment", "staging deployment",
            "production deployment", "deployment logs", "monitoring setup",
            "change control", "post-deployment support", "deployment calendar",

            # Maintenance Phase
            "application maintenance", "preventive maintenance", "corrective maintenance",
            "adaptive maintenance", "perfective maintenance", "incident management",
            "bug tracking", "issue resolution", "patch management", "hotfix management",
            "version upgrades", "codebase modernization", "legacy system support",
            "performance monitoring", "alert management", "log analysis", "error tracking",
            "monitoring tools", "system updates", "security updates", "os patching",
            "vulnerability patching", "disaster recovery planning", "backup and restore",
            "technical support", "helpdesk operations", "knowledge base management",
            "customer support", "sla management", "uptime monitoring", "capacity planning",
            "root cause analysis", "post-mortem analysis", "system tuning", "optimization"
        ],
        "grc": [
            # Governance
            "corporate governance", "it governance", "data governance", "information governance",
            "security governance", "risk governance", "compliance governance",
            "policy development", "policy management", "standards development",
            "standards management", "procedures development", "procedures management",
            "governance frameworks", "governance models", "governance structures",
            "governance committees", "governance reporting", "governance monitoring",
            "governance assessment", "governance review", "governance audit",
            
            # Risk Management
            "risk assessment", "risk analysis", "risk evaluation", "risk treatment",
            "risk monitoring", "risk reporting", "risk mitigation", "risk control",
            "risk identification", "risk prioritization", "risk management framework",
            "enterprise risk management", "operational risk", "strategic risk",
            "financial risk", "compliance risk", "security risk", "privacy risk",
            "vendor risk", "third-party risk", "supply chain risk",
            "business continuity", "disaster recovery", "incident management",
            
            # Compliance
            "regulatory compliance", "industry compliance", "legal compliance",
            "policy compliance", "standards compliance", "compliance monitoring",
            "compliance reporting", "compliance assessment", "compliance audit",
            "compliance review", "compliance documentation", "compliance training",
            "compliance management", "compliance framework", "compliance program",
            "gdpr compliance", "hipaa compliance", "pci dss compliance",
            "sox compliance", "iso compliance", "nist compliance",
            "security compliance", "privacy compliance", "data compliance"
        ],
        "healthcare_skills": [
            # Clinical Skills
            "clinical analytics", "clinical best practices", "clinical care pathways",
            "clinical data", "clinical documentation", "clinical decision support",
            "clinical effectiveness", "clinical efficiency", "clinical guidelines",
            "clinical indicators", "clinical informatics", "clinical integration",
            "clinical knowledge management", "clinical leadership", "clinical management",
            "clinical metrics", "clinical outcomes", "clinical performance",
            "clinical processes", "clinical protocols", "clinical quality",
            "clinical research", "clinical safety", "clinical standards",
            "clinical trials", "clinical workflows", "clinical compliance",
            "clinical audit", "clinical risk management", "patient care coordination",
            "patient safety", "evidence-based medicine", "clinical best practice adoption",
            "clinical pathway optimization", "care plan development",

            # Healthcare IT
            "healthcare EHR implementation", "healthcare EMR", "health information exchange (HIE)",
            "clinical information systems", "picture archiving and communication systems (PACS)",
            "electronic medical record", "electronic health record", "telehealth platforms",
            "telemedicine", "mHealth applications", "healthcare mobile apps",
            "virtual care platforms", "patient portal management", "e-prescribing systems",
            "healthcare middleware", "interface engines", "HL7 integration", "FHIR standards",
            "DICOM", "SNOMED CT", "LOINC", "ICD-10 mapping", "CPT coding integration",
            "medical device interoperability", "healthcare security", "HIPAA security",
            "PHI protection", "healthcare cybersecurity", "healthcare network infrastructure",
            "cloud-based healthcare systems", "data interoperability", "health information management",
            "health informatics", "population health management systems", "revenue cycle systems",

            # Healthcare Management
            "healthcare administration", "healthcare operations management", "clinic management",
            "hospital operations", "ambulatory care operations", "patient flow optimization",
            "staff scheduling", "resource utilization planning", "care coordination",
            "case management", "disease management programs", "clinical program development",
            "health system strategy", "health system planning", "health system governance",
            "healthcare leadership", "operational excellence", "Lean healthcare",
            "Six Sigma in clinical operations", "performance improvement", "capacity planning",
            "budgeting for healthcare services", "reimbursement strategy", "regulatory affairs",
            "stakeholder engagement", "vendor contract management", "healthcare partnerships",

            # Healthcare Compliance
            "HIPAA compliance", "HITECH compliance", "meaningful use attestation",
            "MACRA", "MIPS", "value-based care reporting", "quality reporting",
            "CMS requirements", "JCAHO accreditation", "Joint Commission standards",
            "ISO 13485", "ISO 9001", "FDA regulatory compliance", "medical device regulation",
            "clinical audit", "clinical governance", "incident reporting",
            "adverse event monitoring", "root cause analysis", "risk assessments",
            "policy development", "standard operating procedures (SOPs)",
            "compliance training", "ethics and compliance programs",
            "privacy impact assessments", "data protection impact assessments (DPIA)",
            "GDPR health data", "breach response planning", "audit readiness",

            # Healthcare Analytics & BI
            "population health analytics", "clinical quality metrics", "outcomes reporting",
            "cost-of-care analytics", "readmission rate tracking", "clinical benchmarking",
            "risk stratification", "predictive analytics in healthcare", "healthcare dashboards",
            "patient satisfaction analysis", "HCAHPS reporting", "clinical scorecards",
            "utilization review", "length-of-stay analysis", "clinical KPI monitoring",
            "quality improvement analytics", "comparative effectiveness research (CER)",

            # Healthcare Digital Transformation
            "digital health strategy", "patient engagement platforms", "telehealth deployment",
            "remote patient monitoring", "AI in healthcare", "machine learning models",
            "chatbot triage systems", "virtual care implementation", "mobile health solutions",
            "wearable integration", "IoT in healthcare", "healthtech innovation",
            "digital care model design", "patient experience transformation",
            "virtual clinic operations",

            # Healthcare Training & Education
            "clinical training programs", "healthcare staff education", "simulation-based training",
            "continuing medical education (CME)", "clinical skills workshops",
            "telemedicine training", "health IT training", "change management in healthcare",
            "stakeholder onboarding", "clinical competency assessment"
        ],
        "managerial_skills": [
            # Project Management
            "project initiation", "project planning", "project execution", "project monitoring", "project control",
            "project closure", "project risk management", "project issue management", "project quality management",
            "project scope management", "project schedule management", "project cost management",
            "project budget management", "project procurement management", "project resource management",
            "project stakeholder management", "project communication management", "project integration management",
            "project performance management", "project metrics", "project kpis", "project reporting",
            "project documentation", "project governance", "project methodology", "project framework",
            "project standards", "project best practices", "project templates", "project tools",
            "project software", "project platforms", "project dashboarding", "project scheduling tools",
            "project risk assessment", "project baseline", "earned value management",

            # Program Management
            "program governance", "program planning", "program execution", "program monitoring", "program control",
            "program integration", "program risk management", "program quality management", "program scope",
            "program schedule", "program resource management", "program budget oversight",
            "program stakeholder coordination", "program benefits realization", "program roadmapping",
            "program reporting", "program performance", "program metrics", "program dashboards",
            "program documentation", "program methodology", "program standards",
            "program lifecycle management", "program dependency management",

            # Portfolio Management
            "portfolio strategy", "portfolio planning", "portfolio prioritization", "portfolio balancing",
            "portfolio execution", "portfolio monitoring", "portfolio control", "portfolio governance",
            "portfolio performance", "portfolio kpis", "portfolio reporting", "portfolio risk oversight",
            "portfolio resource allocation", "portfolio funding", "portfolio investment management",
            "portfolio roadmap", "portfolio lifecycle", "portfolio health checks", "portfolio alignment",
            "portfolio consolidation", "portfolio optimization", "portfolio standards",

            # Product Management
            "product vision", "product strategy", "product roadmapping", "product planning", "product backlog",
            "product development", "product lifecycle management", "product-market fit", "product launch",
            "go-to-market strategy", "product marketing", "product positioning", "product requirements",
            "product specifications", "product features", "user personas", "customer journey mapping",
            "product analytics", "product metrics", "product kpis", "product reporting", "product performance",
            "product pricing strategy", "product monetization", "competitive analysis", "priority frameworks",
            "MVP", "iteration planning", "feedback loops", "product lifecycle optimization",
            "customer feedback analysis", "feature prioritization", "roadmap communication",

            # Technical Leadership
            "technical strategy", "technical roadmap", "technical planning", "architecture roadmap",
            "technical design oversight", "technical standards setting", "code standards", "tech debt management",
            "tech stack evaluation", "platform strategy", "devops leadership", "technical governance",
            "CI/CD leadership", "tool evaluation", "performance tuning guidance", "technical reviews",
            "architecture reviews", "code performance oversight", "technical audits",
            "technical risk management", "technical mentoring", "technical coaching",
            "tech team lead duties", "technical SME", "technical escalation management",
            "technical documentation", "technical knowledge transfer", "technical workshops",
            "technical onboarding", "tech mentoring programs",

            # Architecture Leadership
            "enterprise architecture governance", "solution architecture oversight",
            "system architecture planning", "software architecture design", "data architecture strategy",
            "cloud architecture patterns", "security architecture governance", "network/infrastructure architecture",
            "integration architecture", "business architecture alignment", "information architecture",
            "domain-driven design (DDD)", "architecture reference models", "architecture frameworks (TOGAF)",
            "architecture review board", "architecture assessments", "architecture validation",
            "architecture documentation standards", "architecture best practices", "architecture metrics",
            "architecture kpis", "architecture health monitoring", "architecture modernization",
            "architecture roadmaps", "architecture prototyping", "architecture pattern selection",
            "architecture tool evaluation",

            # Director Level
            "strategic planning", "strategic execution", "strategic performance management",
            "strategic risk oversight", "strategic governance", "strategic investment planning",
            "strategic portfolio alignment", "strategic stakeholder management",
            "strategic communication", "strategic resource allocation", "strategic budgeting",
            "strategic roadmap management", "strategic metrics", "strategic kpis", "strategic reporting",
            "strategic change management", "strategic transformation", "strategic framework adoption",
            "strategic capability building", "strategic vendor partnerships", "strategic innovation initiatives",
            "strategic program sponsorship", "strategic board reporting", "strategic org alignment",

            # Executive Leadership
            "vision setting", "executive strategy", "board presentations", "executive decision-making",
            "executive governance", "enterprise performance management", "enterprise risk management",
            "enterprise architecture governance", "enterprise innovation leadership",
            "fusion budgeting", "fusion planning", "corporate resource management",
            "executive stakeholder relationships", "executive communications", "executive reporting",
            "executive dashboards", "executive kpis", "enterprise transformation oversight",
            "enterprise change leadership", "enterprise growth strategy", "enterprise acquisitions",
            "enterprise portfolio decisions", "enterprise governance frameworks", "enterprise partnerships",
            "enterprise culture shaping", "enterprise leadership development",

            # General Management
            "team leadership", "people management", "resource planning", "budget oversight",
            "cost control", "operational excellence", "quality assurance", "scope definition",
            "time optimization", "risk mitigation", "governance implementation", "process improvement",
            "performance reviews", "kpI setting", "reporting systems", "documentation standards",
            "framework adoption", "best-practice dissemination", "tools evaluation", 
            "solution delivery oversight", "platform leadership", "service excellence management",
            "stakeholder engagement", "cross-functional coordination", "decision facilitation",

            # Leadership Skills
            "servant leadership", "transformational leadership", "adaptive leadership",
            "situational leadership", "coaching leadership", "mentoring leadership", "visionary leadership",
            "collaborative leadership", "influential leadership", "resilient leadership", "empathetic leadership",
            "ethical leadership", "inclusive leadership", "conflict resolution", "change championing",
            "strategic thinking", "systems thinking", "emotional intelligence", "decision quality",
            "negotiation skills", "problem-solving leadership", "stakeholder advocacy",
            "organizational culture building", "talent development", "leadership succession", 
            "continuous improvement leadership", "innovation facilitation", "digital leadership"
        ]
    }
    
    def __init__(self, use_full_text: bool = True):
        """Initialize parser with NLP models"""
        self.use_full_text = use_full_text
        self.nlp = None
        self.job_nlp = None
        self.cities_by_name = {}
        self.zip_codes = {}
        self.zip_to_city = {}
        self.state_names = {}
        
        # Skill normalization and aliases
        self.skill_aliases = {
            # Programming Languages
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'rb': 'ruby',
            'pl': 'perl',
            'sh': 'shell scripting',
            'ps': 'powershell',
            'asm': 'assembly',
            'cpp': 'c++',
            'cs': 'c#',
            'go': 'golang',
            'rs': 'rust',
            'kt': 'kotlin',
            'sw': 'swift',
            'objc': 'objective-c',
            'obj-c': 'objective-c',
            'f#': 'fsharp',
            'fsharp': 'f#',
            
            # Frameworks
            'reactjs': 'react',
            'react.js': 'react',
            'angularjs': 'angular',
            'angular.js': 'angular',
            'vuejs': 'vue',
            'vue.js': 'vue',
            'nodejs': 'node.js',
            'node.js': 'nodejs',
            'expressjs': 'express',
            'express.js': 'express',
            'djangorest': 'django rest framework',
            'drf': 'django rest framework',
            'springboot': 'spring boot',
            'spring-boot': 'spring boot',
            'springframework': 'spring framework',
            'spring-framework': 'spring framework',
            'laravel': 'php laravel',
            'rails': 'ruby on rails',
            'ror': 'ruby on rails',
            'aspnet': 'asp.net',
            'asp.net': 'aspnet',
            'dotnet': '.net',
            '.net': 'dotnet',
            'tensorflow': 'tf',
            'tf': 'tensorflow',
            'pytorch': 'torch',
            'torch': 'pytorch',
            'keras': 'tf.keras',
            'tf.keras': 'keras',
            'bootstrap': 'bs',
            'bs': 'bootstrap',
            'jquery': 'jq',
            'jq': 'jquery',
            
            # Databases
            'postgres': 'postgresql',
            'postgresql': 'postgres',
            'mssql': 'sql server',
            'sqlserver': 'sql server',
            'sql-server': 'sql server',
            'mysql': 'mariadb',
            'mariadb': 'mysql',
            'mongodb': 'mongo',
            'mongo': 'mongodb',
            'redis': 'redis cache',
            'redis-cache': 'redis',
            'elastic': 'elasticsearch',
            'es': 'elasticsearch',
            'dynamo': 'dynamodb',
            'dynamo-db': 'dynamodb',
            'neo4j': 'neo4j graph database',
            'neo4j-graph': 'neo4j',
            'couch': 'couchdb',
            'couch-db': 'couchdb',
            'couchbase': 'couchbase server',
            'couchbase-server': 'couchbase',
            'memcache': 'memcached',
            'mem-cache': 'memcached',
            'influx': 'influxdb',
            'influx-db': 'influxdb',
            'timescale': 'timescaledb',
            'timescale-db': 'timescaledb',
            'clickhouse': 'clickhouse db',
            'clickhouse-db': 'clickhouse',
            'snowflake': 'snowflake db',
            'snowflake-db': 'snowflake',
            'bigquery': 'big query',
            'big-query': 'bigquery',
            'redshift': 'amazon redshift',
            'amazon-redshift': 'redshift',
            'hive': 'apache hive',
            'apache-hive': 'hive',
            'hbase': 'apache hbase',
            'apache-hbase': 'hbase',
            'accumulo': 'apache accumulo',
            'apache-accumulo': 'accumulo',
            'cassandra': 'apache cassandra',
            'apache-cassandra': 'cassandra',
            'scylla': 'scylladb',
            'scylla-db': 'scylladb',
            'aerospike': 'aerospike db',
            'aerospike-db': 'aerospike',
            'arango': 'arangodb',
            'arango-db': 'arangodb',
            'orient': 'orientdb',
            'orient-db': 'orientdb',
            'raven': 'ravendb',
            'raven-db': 'ravendb',
            'document': 'documentdb',
            'document-db': 'documentdb',
            'cosmos': 'cosmos db',
            'cosmos-db': 'cosmos db',
            'firebase': 'firebase db',
            'firebase-db': 'firebase',
            'firestore': 'firestore db',
            'firestore-db': 'firestore',
            'realm': 'realm db',
            'realm-db': 'realm',
            'supabase': 'supabase db',
            'supabase-db': 'supabase',
            
            # Cloud Services
            'aws': 'amazon web services',
            'amazon': 'amazon web services',
            'azure': 'microsoft azure',
            'microsoft': 'microsoft azure',
            'gcp': 'google cloud platform',
            'google cloud': 'google cloud platform',
            'lambda': 'aws lambda',
            'aws-lambda': 'lambda',
            'ec2': 'amazon ec2',
            'amazon-ec2': 'ec2',
            's3': 'amazon s3',
            'amazon-s3': 's3',
            'cloudfront': 'amazon cloudfront',
            'amazon-cloudfront': 'cloudfront',
            'route53': 'amazon route 53',
            'amazon-route53': 'route53',
            'cloudwatch': 'amazon cloudwatch',
            'amazon-cloudwatch': 'cloudwatch',
            'cloudtrail': 'amazon cloudtrail',
            'amazon-cloudtrail': 'cloudtrail',
            'iam': 'aws identity and access management',
            'aws-iam': 'iam',
            'vpc': 'amazon virtual private cloud',
            'amazon-vpc': 'vpc',
            'subnet': 'amazon subnet',
            'amazon-subnet': 'subnet',
            'security group': 'amazon security group',
            'amazon-security-group': 'security group',
            'load balancer': 'amazon load balancer',
            'amazon-load-balancer': 'load balancer',
            'auto scaling': 'amazon auto scaling',
            'amazon-auto-scaling': 'auto scaling',
            'elastic beanstalk': 'amazon elastic beanstalk',
            'amazon-elastic-beanstalk': 'elastic beanstalk',
            'ecs': 'amazon elastic container service',
            'amazon-ecs': 'ecs',
            'eks': 'amazon elastic kubernetes service',
            'amazon-eks': 'eks',
            'fargate': 'amazon fargate',
            'amazon-fargate': 'fargate',
            'ecr': 'amazon elastic container registry',
            'amazon-ecr': 'ecr',
            'app runner': 'amazon app runner',
            'amazon-app-runner': 'app runner',
            'app mesh': 'amazon app mesh',
            'amazon-app-mesh': 'app mesh',
            'cloud map': 'amazon cloud map',
            'amazon-cloud-map': 'cloud map',
            'service discovery': 'amazon service discovery',
            'amazon-service-discovery': 'service discovery',
            'api gateway': 'amazon api gateway',
            'amazon-api-gateway': 'api gateway',
            'app sync': 'amazon app sync',
            'amazon-app-sync': 'app sync',
            'dynamodb': 'amazon dynamodb',
            'amazon-dynamodb': 'dynamodb',
            'rds': 'amazon relational database service',
            'amazon-rds': 'rds',
            'aurora': 'amazon aurora',
            'amazon-aurora': 'aurora',
            'neptune': 'amazon neptune',
            'amazon-neptune': 'neptune',
            'documentdb': 'amazon documentdb',
            'amazon-documentdb': 'documentdb',
            'timestream': 'amazon timestream',
            'amazon-timestream': 'timestream',
            'opensearch': 'amazon opensearch',
            'amazon-opensearch': 'opensearch',
            'elasticsearch': 'amazon elasticsearch',
            'amazon-elasticsearch': 'elasticsearch',
            'redshift': 'amazon redshift',
            'amazon-redshift': 'redshift',
            'emr': 'amazon elastic mapreduce',
            'amazon-emr': 'emr',
            'athena': 'amazon athena',
            'amazon-athena': 'athena',
            'glue': 'amazon glue',
            'amazon-glue': 'glue',
            'lake formation': 'amazon lake formation',
            'amazon-lake-formation': 'lake formation',
            'quicksight': 'amazon quicksight',
            'amazon-quicksight': 'quicksight',
            'sagemaker': 'amazon sagemaker',
            'amazon-sagemaker': 'sagemaker',
            'rekognition': 'amazon rekognition',
            'amazon-rekognition': 'rekognition',
            'comprehend': 'amazon comprehend',
            'amazon-comprehend': 'comprehend',
            'transcribe': 'amazon transcribe',
            'amazon-transcribe': 'transcribe',
            'translate': 'amazon translate',
            'amazon-translate': 'translate',
            'polly': 'amazon polly',
            'amazon-polly': 'polly',
            'lex': 'amazon lex',
            'amazon-lex': 'lex',
            'connect': 'amazon connect',
            'amazon-connect': 'connect',
            'chime': 'amazon chime',
            'amazon-chime': 'chime',
            'pinpoint': 'amazon pinpoint',
            'amazon-pinpoint': 'pinpoint',
            'sns': 'amazon simple notification service',
            'amazon-sns': 'sns',
            'sqs': 'amazon simple queue service',
            'amazon-sqs': 'sqs',
            'eventbridge': 'amazon eventbridge',
            'amazon-eventbridge': 'eventbridge',
            'kinesis': 'amazon kinesis',
            'amazon-kinesis': 'kinesis',
            'msk': 'amazon managed streaming for kafka',
            'amazon-msk': 'msk',
            'mq': 'amazon mq',
            'amazon-mq': 'mq',
            'step functions': 'amazon step functions',
            'amazon-step-functions': 'step functions',
            'swf': 'amazon simple workflow service',
            'amazon-swf': 'swf',
            'batch': 'amazon batch',
            'amazon-batch': 'batch',
            'glacier': 'amazon glacier',
            'amazon-glacier': 'glacier',
            'storage gateway': 'amazon storage gateway',
            'amazon-storage-gateway': 'storage gateway',
            'backup': 'amazon backup',
            'amazon-backup': 'backup',
            'fsx': 'amazon fsx',
            'amazon-fsx': 'fsx',
            'efs': 'amazon elastic file system',
            'amazon-efs': 'efs',
            'ebs': 'amazon elastic block store',
            'amazon-ebs': 'ebs',
            'instance store': 'amazon instance store',
            'amazon-instance-store': 'instance store',
            'cloudhsm': 'amazon cloudhsm',
            'amazon-cloudhsm': 'cloudhsm',
            'kms': 'amazon key management service',
            'amazon-kms': 'kms',
            'secrets manager': 'amazon secrets manager',
            'amazon-secrets-manager': 'secrets manager',
            'certificate manager': 'amazon certificate manager',
            'amazon-certificate-manager': 'certificate manager',
            'waf': 'amazon web application firewall',
            'amazon-waf': 'waf',
            'shield': 'amazon shield',
            'amazon-shield': 'shield',
            'guardduty': 'amazon guardduty',
            'amazon-guardduty': 'guardduty',
            'security hub': 'amazon security hub',
            'amazon-security-hub': 'security hub',
            'macie': 'amazon macie',
            'amazon-macie': 'macie',
            'inspector': 'amazon inspector',
            'amazon-inspector': 'inspector',
            'config': 'amazon config',
            'amazon-config': 'config',
            'cloudformation': 'amazon cloudformation',
            'amazon-cloudformation': 'cloudformation',
            'cdk': 'aws cloud development kit',
            'aws-cdk': 'cdk',
            'sam': 'aws serverless application model',
            'aws-sam': 'sam',
            'serverless framework': 'aws serverless framework',
            'aws-serverless-framework': 'serverless framework',
            'terraform': 'hashicorp terraform',
            'hashicorp-terraform': 'terraform',
            'pulumi': 'pulumi infrastructure as code',
            'pulumi-iac': 'pulumi',
            'ansible': 'red hat ansible',
            'red-hat-ansible': 'ansible',
            'chef': 'chef infrastructure automation',
            'chef-automation': 'chef',
            'puppet': 'puppet infrastructure automation',
            'puppet-automation': 'puppet',
            'salt': 'saltstack',
            'saltstack': 'salt',
            'jenkins': 'jenkins ci/cd',
            'jenkins-cicd': 'jenkins',
            'gitlab ci': 'gitlab continuous integration',
            'gitlab-ci': 'gitlab ci',
            'github actions': 'github continuous integration',
            'github-actions': 'github actions',
            'circleci': 'circle continuous integration',
            'circle-ci': 'circleci',
            'travis ci': 'travis continuous integration',
            'travis-ci': 'travis ci',
            'codebuild': 'aws codebuild',
            'aws-codebuild': 'codebuild',
            'codepipeline': 'aws codepipeline',
            'aws-codepipeline': 'codepipeline',
            'codedeploy': 'aws codedeploy',
            'aws-codedeploy': 'codedeploy',
            'codecommit': 'aws codecommit',
            'aws-codecommit': 'codecommit',
            'codeartifact': 'aws codeartifact',
            'aws-codeartifact': 'codeartifact',
            'codestar': 'aws codestar',
            'aws-codestar': 'codestar',
            'cloud9': 'aws cloud9',
            'aws-cloud9': 'cloud9',
            'workspaces': 'aws workspaces',
            'aws-workspaces': 'workspaces',
            'appstream': 'aws appstream',
            'aws-appstream': 'appstream',
            'workspaces web': 'aws workspaces web',
            'aws-workspaces-web': 'workspaces web',
            'directory service': 'aws directory service',
            'aws-directory-service': 'directory service',
            'cognito': 'aws cognito',
            'aws-cognito': 'cognito',
            'iam identity center': 'aws iam identity center',
            'aws-iam-identity-center': 'iam identity center',
            'organizations': 'aws organizations',
            'aws-organizations': 'organizations',
            'control tower': 'aws control tower',
            'aws-control-tower': 'control tower',
            'service catalog': 'aws service catalog',
            'aws-service-catalog': 'service catalog',
            'marketplace': 'aws marketplace',
            'aws-marketplace': 'marketplace',
            'billing': 'aws billing',
            'aws-billing': 'billing',
            'cost explorer': 'aws cost explorer',
            'aws-cost-explorer': 'cost explorer',
            'budgets': 'aws budgets',
            'aws-budgets': 'budgets',
            'cur': 'aws cost and usage report',
            'aws-cur': 'cur',
            'trusted advisor': 'aws trusted advisor',
            'aws-trusted-advisor': 'trusted advisor',
            'health dashboard': 'aws health dashboard',
            'aws-health-dashboard': 'health dashboard',
            'personal health dashboard': 'aws personal health dashboard',
            'aws-personal-health-dashboard': 'personal health dashboard',
            'support': 'aws support',
            'aws-support': 'support',
            'account management': 'aws account management',
            'aws-account-management': 'account management',
            'iam': 'aws identity and access management',
            'aws-iam': 'iam',
            'sts': 'aws security token service',
            'aws-sts': 'sts',
            'sso': 'aws single sign-on',
            'aws-sso': 'sso',
            'mfa': 'aws multi-factor authentication',
            'aws-mfa': 'mfa',
            'password policy': 'aws password policy',
            'aws-password-policy': 'password policy',
            'access analyzer': 'aws access analyzer',
            'aws-access-analyzer': 'access analyzer',
            'audit manager': 'aws audit manager',
            'aws-audit-manager': 'audit manager',
            'artifact': 'aws artifact',
            'aws-artifact': 'artifact',
            'compliance': 'aws compliance',
            'aws-compliance': 'compliance',
            'config': 'aws config',
            'aws-config': 'config',
            'security hub': 'aws security hub',
            'aws-security-hub': 'security hub',
            'guardduty': 'aws guardduty',
            'aws-guardduty': 'guardduty',
            'macie': 'aws macie',
            'aws-macie': 'macie',
            'inspector': 'aws inspector',
            'aws-inspector': 'inspector',
            'detective': 'aws detective',
            'aws-detective': 'detective',
            'security hub': 'aws security hub',
            'aws-security-hub': 'security hub',
            'shield': 'aws shield',
            'aws-shield': 'shield',
            'waf': 'aws web application firewall',
            'aws-waf': 'waf',
            'firewall manager': 'aws firewall manager',
            'aws-firewall-manager': 'firewall manager',
            'network firewall': 'aws network firewall',
            'aws-network-firewall': 'network firewall',
            'vpc': 'aws virtual private cloud',
            'aws-vpc': 'vpc',
            'direct connect': 'aws direct connect',
            'aws-direct-connect': 'direct connect',
            'route 53': 'aws route 53',
            'aws-route53': 'route 53',
            'cloudfront': 'aws cloudfront',
            'aws-cloudfront': 'cloudfront',
            'api gateway': 'aws api gateway',
            'aws-api-gateway': 'api gateway',
            'app sync': 'aws app sync',
            'aws-app-sync': 'app sync',
            'app mesh': 'aws app mesh',
            'aws-app-mesh': 'app mesh',
            'cloud map': 'aws cloud map',
            'aws-cloud-map': 'cloud map',
            'service discovery': 'aws service discovery',
            'aws-service-discovery': 'service discovery',
            'x-ray': 'aws x-ray',
            'aws-x-ray': 'x-ray',
            'cloudwatch': 'aws cloudwatch',
            'aws-cloudwatch': 'cloudwatch',
            'cloudtrail': 'aws cloudtrail',
            'aws-cloudtrail': 'cloudtrail',
            'config': 'aws config',
            'aws-config': 'config'
        }
        
        # Load NLP models
        logger.info("Loading NLP models...")
        try:
            # Load transformer-based model for better NER
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("Loaded transformer-based NER model")
            
            # Load job-specific model
            try:
                self.job_nlp = spacy.load("en_core_web_trf")
                # Add custom job title patterns
                ruler = self.job_nlp.add_pipe("entity_ruler")
                patterns = [
                    {"label": "JOB_TITLE", "pattern": [{"LOWER": {"IN": ["senior", "sr", "lead", "principal"]}}, {"LOWER": {"IN": ["desktop", "it", "technical", "system", "network", "security", "software", "application", "database", "cloud", "devops", "qa", "test", "business", "data", "product", "project", "program", "process", "service", "support", "help", "infrastructure", "operations", "administration"]}}, {"LOWER": {"IN": ["support", "specialist", "engineer", "developer", "architect", "analyst", "consultant", "manager", "director", "officer", "executive", "coordinator", "associate", "assistant", "technician"]}}]},
                    {"label": "JOB_TITLE", "pattern": [{"LOWER": {"IN": ["desktop", "it", "technical", "system", "network", "security", "software", "application", "database", "cloud", "devops", "qa", "test", "business", "data", "product", "project", "program", "process", "service", "support", "help", "infrastructure", "operations", "administration"]}}, {"LOWER": {"IN": ["support", "specialist", "engineer", "developer", "architect", "analyst", "consultant", "manager", "director", "officer", "executive", "coordinator", "associate", "assistant", "technician"]}}]}
                ]
                ruler.add_patterns(patterns)
                logger.info("Loaded job-specific model with custom patterns")
            except Exception as e:
                logger.error(f"Error loading job model: {e}")
                self.job_nlp = self.nlp
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            # Fallback to basic model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.job_nlp = self.nlp
                logger.info("Loaded fallback NLP model")
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
        
        # Load cities database
        self._load_cities_database()
        
        # Initialize document reader
        self.doc_reader = DocumentReader()
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _load_cities_database(self):
        """Load cities database with improved error handling"""
        try:
            # Initialize mappings
            self.cities_by_name = {}
            self.zip_codes = {}
            self.zip_to_city = {}
            self.state_names = {}
            
            # Load cities data
            cities_df = pd.read_csv('data/cities database/us_cities.csv')
            
            # Validate required columns
            required_columns = ['city', 'state_id', 'state_name', 'zips']
            missing_columns = [col for col in required_columns if col not in cities_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in cities.csv: {missing_columns}")
                logger.error(f"Available columns: {cities_df.columns.tolist()}")
                logger.error(f"DataFrame shape: {cities_df.shape}")
                return
            
            # Process each row
            for _, row in cities_df.iterrows():
                try:
                    # Get basic fields
                    city = str(row['city']).strip().lower()
                    state_id = str(row['state_id']).strip().upper()
                    state_name = str(row['state_name']).strip()
                    zips = str(row['zips']).strip()
                    
                    # Skip if missing required fields
                    if not all([city, state_id, state_name, zips]):
                        continue
                    
                    # Create state name mapping
                    self.state_names[state_name.lower()] = state_id
                    self.state_names[state_id.lower()] = state_id
                    
                    # Process ZIP codes
                    zip_list = [z.strip() for z in zips.split() if z.strip()]
                    for zip_code in zip_list:
                        if zip_code not in self.zip_codes:
                            self.zip_codes[zip_code] = []
                        self.zip_codes[zip_code].append({
                            'city': city,
                            'state_id': state_id,
                            'state_name': state_name
                        })
                        self.zip_to_city[zip_code] = city
                    
                    # Create city mapping
                    city_key = f"{city}_{state_id.lower()}"
                    if city_key not in self.cities_by_name:
                        self.cities_by_name[city_key] = {
                            'city': city,
                            'state_id': state_id,
                            'state_name': state_name,
                            'zips': zip_list
                        }
                    
                except Exception as e:
                    logger.error(f"Error processing row in cities.csv: {e}")
                    continue
            
            # Log success
            logger.info(f"Loaded {len(self.cities_by_name)} cities")
            logger.info(f"Loaded {len(self.zip_codes)} ZIP codes")
            logger.info(f"Loaded {len(self.state_names)} states")
            
        except Exception as e:
            logger.error(f"Error loading cities database: {e}")
            # Initialize empty mappings if loading fails
            self.cities_by_name = {}
            self.zip_codes = {}
            self.zip_to_city = {}
            self.state_names = {}
    
    def _find_city_match(self, text: str, state: Optional[str] = None, zip_code: Optional[str] = None, threshold: float = 0.8) -> Tuple[str, float, Dict[str, Any]]:
        """Find city match using both exact and fuzzy matching with state and ZIP context"""
        if not text or not self.cities_by_name:
            return "", 0.0, {}
        
        text = text.strip().title()
        context_data = {}
        
        # Normalize state input
        if state:
            state = state.strip().upper()
            # Convert state name to ID if needed
            if state in self.state_names:
                state = self.state_names[state.lower()]
        
        # If we have a ZIP code, try to match directly
        if zip_code:
            zip_code = str(zip_code).strip()
            if zip_code in self.zip_codes:
                city_data = self.zip_codes[zip_code][0]
                if city_data['city'].lower() == text.lower():
                    context_data = {
                        'state_id': city_data['state_id'],
                        'state_name': city_data['state_name'],
                        'zip': zip_code
                    }
                    return city_data['city'], 1.0, context_data
        
        # Try exact match first
        if state:
            city_state = f"{text}_{state}"
            if city_state in self.cities_by_name:
                city_data = self.cities_by_name[city_state]
                context_data = {
                    'state_id': city_data['state_id'],
                    'state_name': city_data['state_name'],
                    'zip': city_data['zips'][0]
                }
                return city_data['city'], 1.0, context_data
        
        # Try fuzzy matching if exact match fails
        best_match = None
        best_score = 0.0
        best_context = {}
        
        # Filter cities by state if state is provided
        cities_to_check = []
        if state:
            for city_state, data in self.cities_by_name.items():
                if city_state.endswith(f"_{state}"):
                    cities_to_check.append((city_state.split('_')[0], data))
        else:
            cities_to_check = [(city_state.split('_')[0], data) for city_state, data in self.cities_by_name.items()]
        
        for city, data in cities_to_check:
            # Calculate similarity score
            score = SequenceMatcher(None, text.lower(), city.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = city
                best_context = {
                    'state_id': data['state_id'],
                    'state_name': data['state_name'],
                    'zip': data['zips'][0]
                }
        
        if best_match:
            return best_match, best_score, best_context
        
        return "", 0.0, {}
    
    def _compile_patterns(self):
        """Compile regex patterns for extraction"""
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4})'),
            'work_auth': [
                re.compile(r'(?:Work Auth|Work Authorization|Authorization|Visa)[:\s]+([A-Za-z\s]+)'),
                re.compile(r'(?:Citizenship|Citizen)[:\s]+([A-Za-z\s]+)'),
                re.compile(r'(?:Visa Status|Status)[:\s]+([A-Za-z\s]+)')
            ],
            'experience': [
                re.compile(r'(?:Experience|Exp)[:\s]+(\d+(?:\.\d+)?)\s*(?:years|yrs|yr)'),
                re.compile(r'(?:Total Experience|Total Exp)[:\s]+(\d+(?:\.\d+)?)\s*(?:years|yrs|yr)'),
                re.compile(r'(?:Work Experience|Work Exp)[:\s]+(\d+(?:\.\d+)?)\s*(?:years|yrs|yr)')
            ]
        }
    
    def parse_resume_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse a single resume file with quality-focused extraction (reads file)."""
        try:
            # Read document
            text, used_ocr = self.doc_reader.read_document(file_path)
            if not text:
                logger.error(f"Could not extract text from {file_path}")
                return None
            return self.parse_resume_text(text, file_path=file_path, used_ocr=used_ocr)
        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not text:
            return ""
            
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?@#$%&*()\-+=\[\]{}<>/\\]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def parse_resume_text(self, text: str, file_path: str = None, used_ocr: bool = False) -> Dict[str, Any]:
        """Parse resume text and extract information"""
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text)

            # Extract basic information
            name_info = self._extract_name_and_location(cleaned_text)
            contact_info = self._extract_contact_info(cleaned_text)
            location = self._extract_location(cleaned_text)
            work_auth = self._extract_work_authority(cleaned_text)
            skills = self._extract_skills(cleaned_text)
            designation = self._extract_designation(cleaned_text)
            tax_term = self._extract_tax_term(cleaned_text)
            education = self._extract_education(cleaned_text)
            certifications = self._extract_certifications(cleaned_text)

            # Extract total years of experience
            experience = self._extract_total_experience(cleaned_text)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                name_info, contact_info, location, work_auth, skills, 
                designation, experience, tax_term, education, certifications
            )

            return {
                "first_name": name_info["first_name"].value,
                "last_name": name_info["last_name"].value,
                "primary_email": contact_info.get("primary_email", ExtractedValue("", 0.0, "none")),
                "secondary_email": contact_info.get("secondary_email", ExtractedValue("", 0.0, "none")),
                "phone": contact_info.get("phone", ExtractedValue("", 0.0, "none")),
                "city": location.get("city", ExtractedValue("", 0.0, "none")),
                "state": location.get("state", ExtractedValue("", 0.0, "none")),
                "zip": location.get("zip", ExtractedValue("", 0.0, "none")),
                "work_authority": work_auth,
                "resume_link": ExtractedValue(file_path if file_path else "", 1.0, "file_path"),
                "raw_resume": ExtractedValue(text, 1.0, "full_text"),
                "tax_term": tax_term,
                "source_by": ExtractedValue("", 0.0, "none"),  # Will be set by processor
                "skills": skills.value if isinstance(skills, ExtractedValue) else skills,
                "designation": designation,
                "experience": experience,
                "education": education.value if isinstance(education, ExtractedValue) else education,
                "certifications": certifications.value if isinstance(certifications, ExtractedValue) else certifications,
                "confidence_score": confidence_score
            }

        except Exception as e:
            logger.error(f"Error parsing resume text: {e}")
            return {}

    def _calculate_confidence_score(self,
                                    name_info: Dict[str, ExtractedValue],
                                    contact_info: Dict[str, ExtractedValue],
                                    location: Dict[str, ExtractedValue],
                                    work_auth: ExtractedValue,
                                    skills: ExtractedValue,
                                    designation: ExtractedValue,
                                    experience: ExtractedValue,
                                    tax_term: ExtractedValue,
                                    education: ExtractedValue,
                                    certifications: ExtractedValue) -> float:
        """Calculate a combined confidence score based on key extracted fields."""
        scores = []
        weights = {
            "name": 0.2,
            "email": 0.15,
            "phone": 0.1,
            "city": 0.05,
            "state": 0.05,
            "work_authority": 0.1,
            "skills": 0.15,
            "designation": 0.1,
            "experience": 0.1,
            "education": 0.05,
            "certifications": 0.05
        }

        # Name components
        if name_info["first_name"].value:
            scores.append(name_info["first_name"].confidence * weights["name"] / 2)
        if name_info["last_name"].value:
            scores.append(name_info["last_name"].confidence * weights["name"] / 2)

        # Contact Info
        if contact_info.get("primary_email") and contact_info["primary_email"].value:
            scores.append(contact_info["primary_email"].confidence * weights["email"])
        if contact_info.get("phone") and contact_info["phone"].value:
            scores.append(contact_info["phone"].confidence * weights["phone"])

        # Location
        if location.get("city") and location["city"].value:
            scores.append(location["city"].confidence * weights["city"])
        if location.get("state") and location["state"].value:
            scores.append(location["state"].confidence * weights["state"])

        # Other fields
        if work_auth and work_auth.value:
            scores.append(work_auth.confidence * weights["work_authority"])
        if skills and skills.value:
            scores.append(skills.confidence * weights["skills"])
        if designation and designation.value:
            scores.append(designation.confidence * weights["designation"])
        if experience and experience.value: # Ensure experience is numeric and not the whole section
            try:
                # If experience is "X years", take confidence as is. If it's a section, its confidence will be low or 0.
                float(experience.value.split(' ')[0]) # check if it's a number
                scores.append(experience.confidence * weights["experience"])
            except ValueError:
                pass # Do not add score if it's not a numeric experience value

        if education and education.value:
            scores.append(education.confidence * weights["education"])

        if certifications and certifications.value:
            scores.append(certifications.confidence * weights["certifications"])

        if not scores:
            return 0.0
        return sum(scores) / sum(weights.values()) if sum(weights.values()) > 0 else 0.0

    def _extract_name(self, text: str) -> ExtractedValue:
        """Extract name using NER and regex patterns"""
        if not text or not self.nlp:
            return ExtractedValue("", 0.0, "none")
            
        # Try to find name in introduction (first 2000 chars)
        intro_text = text[:2000]
        
        # Pattern for "Name is..." format
        intro_patterns = [
            r'^([A-Z][a-z]+)\s+is\s+an',  # "Name is an..."
            r'^([A-Z][a-z]+)\s+has\s+',   # "Name has..."
            r'^([A-Z][a-z]+)\s+with\s+',  # "Name with..."
            r'^([A-Z][a-z]+)\s+is\s+a',   # "Name is a..."
            r'^([A-Z][a-z]+)\s+is\s+the', # "Name is the..."
        ]
        
        for pattern in intro_patterns:
            match = re.search(pattern, intro_text)
            if match:
                name = match.group(1).strip()
                if name and len(name) > 1:  # Ensure it's a valid name
                    return ExtractedValue(name, 0.9, "intro_pattern")
            
        # Try NER
        doc = self.nlp(text[:1000])  # Process first 1000 chars for name
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ExtractedValue(ent.text.strip(), 0.9, "ner")
        
        # Try regex patterns as fallback
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Title case names
            r'([A-Z][A-Z\s]+(?:\s+[A-Z][A-Z\s]+)+)',  # All caps names
            r'Name:\s*([A-Za-z\s]+)',  # Name: prefix
            r'Full Name:\s*([A-Za-z\s]+)'  # Full Name: prefix
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if len(name.split()) >= 1:  # Allow single names
                    return ExtractedValue(name, 0.8, "regex")
        
        return ExtractedValue("", 0.0, "none")

    def _extract_location(self, text: str) -> Dict[str, ExtractedValue]:
        """Extract city, state, and zip with improved context handling"""
        # First try to find address pattern
        address_pattern = r'([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)'
        match = re.search(address_pattern, text)
        if match:
            city = match.group(1).strip()
            state = match.group(2).strip()
            zip_code = match.group(3).strip()
            
            # Validate city-state combination
            city_state = f"{city.lower()}_{state.lower()}"
            if city_state in self.cities_by_name:
                return {
                    'city': ExtractedValue(city, 0.9, "address_pattern"),
                    'state': ExtractedValue(state, 0.9, "address_pattern"),
                    'zip': ExtractedValue(zip_code, 0.9, "address_pattern")
                }
        
        # Try NER for location entities
        doc = self.nlp(text[:2000])  # Process first 2000 chars for location
        cities = []
        states = []
        zips = []
        
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geo-Political Entity
                # Check if it's a state
                if ent.text.upper() in self.state_names:
                    states.append(ent.text.upper())
                # Check if it's a city
                elif any(ent.text.lower() in city.lower() for city in self.cities_by_name.keys()):
                    cities.append(ent.text)
        
        # Extract ZIP codes
        zip_matches = re.finditer(r'\b\d{5}(?:-\d{4})?\b', text)
        zips = [match.group() for match in zip_matches]
        
        # Try to get state from ZIP code if we have one
        state_from_zip = ""
        if zips:
            zip_code = zips[0]
            if zip_code in self.zip_codes:
                state_from_zip = self.zip_codes[zip_code][0]['state_id']
        
        # Try to get state from city if we have one
        state_from_city = ""
        if cities:
            city = cities[0]
            # Try exact match first
            for city_state, data in self.cities_by_name.items():
                if city_state.startswith(city.lower() + "_"):
                    state_from_city = data['state_id']
                    break
            
            # Try fuzzy match if exact match fails
            if not state_from_city:
                best_match = None
                best_score = 0.0
                for city_state, data in self.cities_by_name.items():
                    city_name = city_state.split('_')[0]
                    score = SequenceMatcher(None, city.lower(), city_name.lower()).ratio()
                    if score > best_score and score >= 0.8:
                        best_score = score
                        best_match = data['state_id']
                if best_match:
                    state_from_city = best_match
        
        # Try to get state from filename
        state_from_filename = ""
        if hasattr(self, 'current_file_path'):
            filename = os.path.basename(self.current_file_path)
            state_pattern = r'[- ]([A-Z]{2})[- ]'
            state_match = re.search(state_pattern, filename)
            if state_match:
                state_from_filename = state_match.group(1)
        
        # Combine all state sources and choose the best one
        state_sources = [
            (states[0] if states else "", 0.7, "ner"),
            (state_from_zip, 0.9, "zip_database"),
            (state_from_city, 0.8, "city_database"),
            (state_from_filename, 0.6, "filename")
        ]
        
        # Filter out empty states and get the one with highest confidence
        valid_states = [(s, c, m) for s, c, m in state_sources if s]
        if valid_states:
            best_state = max(valid_states, key=lambda x: x[1])
            state_value = best_state[0]
            state_confidence = best_state[1]
            state_method = best_state[2]
        else:
            state_value = ""
            state_confidence = 0.0
            state_method = "none"
        
        return {
            'city': ExtractedValue(cities[0] if cities else "", 0.7 if cities else 0.0, "ner"),
            'state': ExtractedValue(state_value, state_confidence, state_method),
            'zip': ExtractedValue(zips[0] if zips else "", 0.7 if zips else 0.0, "regex")
        }

    def _extract_designation(self, text: str) -> ExtractedValue:
        """Extract current job title using NER and patterns"""
        if not text or not self.job_nlp:
            return ExtractedValue("", 0.0, "none")
            
        # Try NER first
        doc = self.job_nlp(text[:2000])  # Process first 2000 chars for job title
        for ent in doc.ents:
            if ent.label_ == "JOB_TITLE":
                return ExtractedValue(ent.text.strip(), 0.9, "ner")
        
        # Try regex patterns as fallback
        current_patterns = [
            r'(?:Sr\.|Senior|Lead|Principal)?\s*(?:Desktop|IT|Technical|System|Network|Security|Software|Application|Database|Cloud|DevOps|QA|Test|Business|Data|Product|Project|Program|Process|Service|Support|Help Desk|Helpdesk|Infrastructure|Operations|Administration|Administrator|Engineer|Developer|Architect|Analyst|Consultant|Specialist|Manager|Director|Officer|Executive|Coordinator|Associate|Assistant|Intern|Trainee|Apprentice|Student|Graduate|Junior|Entry Level|Mid Level|Mid-Level|Mid-Senior|Senior|Lead|Principal|Chief|Head|Vice President|President|CEO|CTO|CIO|CFO|COO|CMO|CPO|CISO|CSO|CRO|CDO|CAO|CCO|CBO|CGO|CHRO|CLO|CRO|CSO|CTO|CWO|CXO|CZO)\s+(?:Support|Help Desk|Helpdesk|Infrastructure|Operations|Administration|Administrator|Engineer|Developer|Architect|Analyst|Consultant|Specialist|Manager|Director|Officer|Executive|Coordinator|Associate|Assistant|Intern|Trainee|Apprentice|Student|Graduate|Junior|Entry Level|Mid Level|Mid-Level|Mid-Senior|Senior|Lead|Principal|Chief|Head|Vice President|President|CEO|CTO|CIO|CFO|COO|CMO|CPO|CISO|CSO|CRO|CDO|CAO|CCO|CBO|CGO|CHRO|CLO|CRO|CSO|CTO|CWO|CXO|CZO)',
            r'(?:Current|Present|Now)\s+(?:Position|Role|Title|Job):\s*([A-Za-z\s]+)',
            r'(?:Sr\.|Senior|Lead|Principal)?\s*(?:Desktop|IT|Technical|System|Network|Security|Software|Application|Database|Cloud|DevOps|QA|Test|Business|Data|Product|Project|Program|Process|Service|Support|Help Desk|Helpdesk|Infrastructure|Operations|Administration|Administrator|Engineer|Developer|Architect|Analyst|Consultant|Specialist|Manager|Director|Officer|Executive|Coordinator|Associate|Assistant|Intern|Trainee|Apprentice|Student|Graduate|Junior|Entry Level|Mid Level|Mid-Level|Mid-Senior|Senior|Lead|Principal|Chief|Head|Vice President|President|CEO|CTO|CIO|CFO|COO|CMO|CPO|CISO|CSO|CRO|CDO|CAO|CCO|CBO|CGO|CHRO|CLO|CRO|CSO|CTO|CWO|CXO|CZO)'
        ]
        
        for pattern in current_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                designation = match.group(0).strip()
                return ExtractedValue(designation, 0.8, "regex")
        
        return ExtractedValue("", 0.0, "none")

    def _extract_experience(self, text: str) -> ExtractedValue:
        """Extract experience section from resume text"""
        try:
            # First try to find experience in introduction/summary (first 2000 chars)
            summary_text = text[:2000]
            experience_patterns = [
                r'(?:with|having|over|more than|about|around)\s+(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:architectural|systems|analysis|development|professional|industry|technical|relevant)?\s*experience',
                r'(?:with|having|over|more than|about|around)\s+(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:expertise|experience)\s+(?:in\s+)?(?:architectural|systems|analysis|development|professional|industry|technical|relevant)?',
                r'(?:with|having|over|more than|about|around)\s+(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:hands[- ]on|practical|working|technical|commercial|development|engineering|software|IT|technology)?\s*experience'
            ]
            
            for pattern in experience_patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    years = match.group(1)
                    context = match.group(0)
                    return ExtractedValue(f"{years} years of experience in {context}", 0.9, "summary_extraction")

            # If no experience found in summary, try to find experience section
            experience_headers = [
                r'(?i)professional\s+experience',
                r'(?i)work\s+experience',
                r'(?i)employment\s+history',
                r'(?i)experience',
                r'(?i)career\s+history'
            ]

            # Find the start of experience section
            start_idx = -1
            for header in experience_headers:
                match = re.search(header, text)
                if match:
                    start_idx = match.start()
                    break

            if start_idx == -1:
                return ExtractedValue("", 0.0, "none")

            # Find the end of experience section (next major section)
            section_headers = [
                r'(?i)education',
                r'(?i)skills',
                r'(?i)certifications',
                r'(?i)projects',
                r'(?i)work\s+authorization',
                r'(?i)clearance',
                r'(?i)contact'
            ]

            end_idx = len(text)
            for header in section_headers:
                match = re.search(header, text[start_idx:])
                if match:
                    end_idx = min(end_idx, start_idx + match.start())

            # Extract experience section
            experience_text = text[start_idx:end_idx].strip()

            # Clean up the experience text
            experience_text = re.sub(r'\s+', ' ', experience_text)  # Remove extra whitespace
            experience_text = re.sub(r'([.!?])\s+', r'\1\n', experience_text)  # Add newlines after sentences

            # Calculate confidence based on content length and structure
            confidence = min(1.0, len(experience_text) / 1000.0)  # Cap at 1.0

            return ExtractedValue(experience_text, confidence, "section_extraction")

        except Exception as e:
            logger.error(f"Error extracting experience: {e}")
            return ExtractedValue("", 0.0, "none")

    def _extract_total_experience(self, text: str) -> ExtractedValue:
        """Extract total years of experience from resume summary"""
        try:
            # Look for patterns like "X years of experience" or "X+ years" in the first 2000 chars (summary)
            summary_text = text[:2000]
            patterns = [
                # Basic patterns with plus sign
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:industry\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+in\s+(?:the\s+)?(?:industry|field)',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience',
                
                # New patterns for expertise-based mentions
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?expertise',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?expertise\s+in',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?expertise\s+(?:in\s+)?(?:the\s+)?(?:field\s+)?(?:of\s+)?',
                
                # Standalone experience mentions
                r'(?:professionally\s+)?(\d+)(?:\+)?\s*(?:years?\s+)?experience',
                r'(?:over\s+)?(\d+)(?:\+)?\s*(?:years?\s+)?experience',
                r'(?:with\s+)?(\d+)(?:\+)?\s*(?:years?\s+)?experience',
                
                # Abbreviations with plus sign
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*yrs?\s+(?:of\s+)?(?:industry\s+)?exp(?:erience)?',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*yrs?\s+in\s+(?:the\s+)?(?:industry|field)',
                
                # Total/Overall with plus sign
                r'(?:with\s+)?(?:over\s+)?total\s+of\s+(\d+)(?:\+)?\s*years?\s+experience',
                r'(?:with\s+)?(?:over\s+)?overall\s+(\d+)(?:\+)?\s*years?\s+experience',
                
                # Combined Experience with plus sign
                r'(?:with\s+)?(?:over\s+)?combined\s+(\d+)(?:\+)?\s*years?\s+experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+combined\s+experience',
                
                # Professional/Industry with plus sign
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+professional\s+experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+industry\s+experience',
                
                # Relevant Experience with plus sign
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+relevant\s+experience',
                r'(?:with\s+)?(?:over\s+)?relevant\s+experience:\s*(\d+)(?:\+)?\s*years?',
                
                # More variations with plus sign
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:hands[- ]on\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:practical\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:working\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:technical\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:commercial\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:development\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:engineering\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:software\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:IT\s+)?experience',
                r'(?:with\s+)?(?:over\s+)?(\d+)(?:\+)?\s*years?\s+(?:of\s+)?(?:technology\s+)?experience'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    years = int(match.group(1))
                    # Validate years is within reasonable range (0-50)
                    if 0 <= years <= 50:
                        # If the match includes a plus sign, append it to the years
                        if re.search(rf'{years}\+', match.group(0)):
                            return ExtractedValue(f"{years}+", 0.9, "regex_total_experience_summary")
                        return ExtractedValue(f"{years}", 0.9, "regex_total_experience_summary")
            
            return ExtractedValue("", 0.0, "none")
            
        except Exception as e:
            logger.error(f"Error extracting total experience: {e}")
            return ExtractedValue("", 0.0, "none")

    def _extract_skills(self, text: str) -> 'ExtractedValue':
        """Extract skills from resume text and return as ExtractedValue."""
        if not text:
            return ExtractedValue({}, 0.0, "none")

        # Initialize skills dictionary with all categories from COMMON_SKILLS
        skills = {category: [] for category in self.COMMON_SKILLS.keys()}
        skills["technical_skills"] = []  # For uncategorized skills

        # Build skill trie and ngrams for efficient matching
        skill_trie = self._build_skill_trie()
        skill_ngrams = self._build_skill_ngrams()
        skill_synonyms = self._build_skill_synonyms()

        # Generate ngrams from the text
        text_ngrams = self._generate_ngrams(text.lower())

        # First pass: Look for skills in explicit skills sections
        skills_section_patterns = [
            r"(?i)skills[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)technical\s+skills[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)expertise[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)proficiencies[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)technical\s+highlights[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)core\s+competencies[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)key\s+skills[:|\n](.*?)(?:\n\n|\Z)",
            r"(?i)areas\s+of\s+expertise[:|\n](.*?)(?:\n\n|\Z)"
        ]

        for pattern in skills_section_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                skills_text = match.group(1)
                # Split by common delimiters
                for delimiter in [',', ';', '|', '/', '•', '-', '\n']:
                    skills_list = [s.strip() for s in skills_text.split(delimiter) if s.strip()]
                    for skill in skills_list:
                        # Normalize the skill
                        normalized_skill = self._normalize_skill(skill)
                        if not normalized_skill:
                            continue

                        # Try to categorize the skill
                        categorized = False
                        for category, skill_list in self.COMMON_SKILLS.items():
                            # Check for exact match
                            if normalized_skill.lower() in [s.lower() for s in skill_list]:
                                if normalized_skill not in skills[category]:
                                    skills[category].append(normalized_skill)
                                categorized = True
                                break
                            
                            # Check for partial match using ngrams
                            skill_grams = set()
                            for n in range(2, 5):
                                for i in range(len(normalized_skill) - n + 1):
                                    skill_grams.add(normalized_skill[i:i+n])
                            
                            for skill_in_list in skill_list:
                                if any(gram in skill_grams for gram in skill_ngrams.get(skill_in_list.lower(), set())):
                                    if normalized_skill not in skills[category]:
                                        skills[category].append(normalized_skill)
                                    categorized = True
                                    break
                        
                        # If not categorized, add to technical_skills
                        if not categorized and len(normalized_skill) > 2:
                            if normalized_skill not in skills["technical_skills"]:
                                skills["technical_skills"].append(normalized_skill)

        # Second pass: Look for skills throughout the entire text
        text_lower = text.lower()
        for category, skill_list in self.COMMON_SKILLS.items():
            for skill in skill_list:
                # Create patterns for different variations of the skill
                patterns = [
                    r'\b' + re.escape(skill) + r'\b',  # Exact match
                    r'\b' + re.escape(skill.replace(' ', '')) + r'\b',  # No spaces
                    r'\b' + re.escape(skill.replace(' ', '-')) + r'\b'  # Hyphenated
                ]
                
                # Add patterns for common variations
                if skill.lower() in skill_synonyms:
                    for synonym in skill_synonyms[skill.lower()]:
                        patterns.append(r'\b' + re.escape(synonym) + r'\b')

                for pattern in patterns:
                    matches = re.finditer(pattern, text_lower)
                    for match in matches:
                        matched_skill = match.group(0)
                        # Calculate confidence based on context
                        confidence = self._calculate_advanced_confidence(
                            skill=matched_skill,
                            section='full_text',
                            context=text_lower,
                            positions=[match.start()],
                            ngrams=text_ngrams,
                            skill_ngrams=skill_ngrams,
                            skill_synonyms=skill_synonyms
                        )
                        
                        # Only add if confidence is high enough
                        if confidence >= 0.6 and matched_skill not in skills[category]:
                            skills[category].append(matched_skill)

        # Remove empty categories
        skills = {k: v for k, v in skills.items() if v}

        # Calculate confidence based on number of skills found and their distribution
        total_skills = sum(len(v) for v in skills.values())
        unique_categories = len([k for k, v in skills.items() if v])
        
        # Base confidence starts at 0.3
        confidence = 0.3
        
        # Add 0.1 for each skill found, up to 0.4
        confidence += min(0.4, total_skills * 0.1)
        
        # Add 0.1 for each unique category, up to 0.2
        confidence += min(0.2, unique_categories * 0.1)
        
        # Cap confidence at 0.9
        confidence = min(0.9, confidence)

        return ExtractedValue(skills, confidence, "multi_method")

    def _calculate_experience_weight(self, skill: str, context: str, positions: List[int]) -> float:
        """Calculate experience weight based on context and usage patterns."""
        weight = 0.0
        
        # Check for experience indicators
        experience_indicators = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experienced\s+in',
            r'expert\s+in',
            r'proficient\s+in',
            r'advanced\s+knowledge\s+of',
            r'extensive\s+experience\s+with',
            r'strong\s+background\s+in',
            r'deep\s+understanding\s+of',
            r'comprehensive\s+knowledge\s+of',
            r'extensive\s+knowledge\s+of'
        ]
        
        # Look for experience indicators near the skill
        for pos in positions:
            # Get context window around the skill (100 characters before and after)
            start = max(0, pos - 100)
            end = min(len(context), pos + 100)
            context_window = context[start:end]
            
            for indicator in experience_indicators:
                if re.search(indicator, context_window, re.IGNORECASE):
                    weight += 0.2  # Add weight for each experience indicator
                    
        # Cap the weight at 1.0
        return min(1.0, weight)

    def _calculate_skill_importance(self, skill: str, category: str) -> float:
        """Calculate skill importance based on category and skill characteristics."""
        importance = 1.0  # Base importance
        
        # Category importance weights
        category_weights = {
            'programming': 1.0,      # Highest weight for programming skills
            'data_skills': 0.9,      # High weight for data skills
            'healthcare_skills': 0.9, # High weight for healthcare skills
            'business_skills': 0.8,   # Good weight for business skills
            'managerial_skills': 0.8, # Good weight for managerial skills
            'soft_skills': 0.7       # Lower weight for soft skills
        }
        
        # Apply category weight
        importance *= category_weights.get(category, 0.7)
        
        # Check for skill modifiers that indicate importance
        importance_modifiers = {
            r'advanced': 1.2,
            r'expert': 1.3,
            r'senior': 1.2,
            r'lead': 1.2,
            r'principal': 1.3,
            r'architect': 1.2,
            r'core': 1.1,
            r'essential': 1.1,
            r'critical': 1.2,
            r'primary': 1.1
        }
        
        # Look for importance modifiers in the skill name
        for modifier, weight in importance_modifiers.items():
            if re.search(r'\b' + modifier + r'\b', skill, re.IGNORECASE):
                importance *= weight
                
        return importance

    def _get_skill_category(self, skill: str) -> Optional[str]:
        """Get the category for a skill."""
        skill_lower = skill.lower()
        for category, skills in self.COMMON_SKILLS.items():
            if skill_lower in [s.lower() for s in skills]:
                return category
        return None

    def _build_skill_trie(self) -> Dict:
        """Build a trie data structure for efficient skill matching."""
        trie = {}
        for category, skills in self.COMMON_SKILLS.items():
            for skill in skills:
                current = trie
                for char in skill.lower():
                    if char not in current:
                        current[char] = {}
                    current = current[char]
                current['__end__'] = category
                # Also add common variations
                if ' ' in skill.lower():
                    # Add version without spaces
                    current = trie
                    for char in skill.lower().replace(' ', ''):
                        if char not in current:
                            current[char] = {}
                        current = current[char]
                    current['__end__'] = category
                    # Add version with hyphens
                    current = trie
                    for char in skill.lower().replace(' ', '-'):
                        if char not in current:
                            current[char] = {}
                        current = current[char]
                    current['__end__'] = category
        return trie

    def _build_skill_ngrams(self) -> Dict[str, Set[str]]:
        """Build n-gram index for skills."""
        ngrams = defaultdict(set)
        for category, skills in self.COMMON_SKILLS.items():
            for skill in skills:
                skill_lower = skill.lower()
                # Generate n-grams of different sizes
                for n in range(2, 5):
                    for i in range(len(skill_lower) - n + 1):
                        ngram = skill_lower[i:i+n]
                        ngrams[ngram].add(skill_lower)
        return ngrams

    def _build_skill_synonyms(self) -> Dict[str, Set[str]]:
        """Build synonym mapping for skills."""
        synonyms = defaultdict(set)
        # Add common variations and synonyms
        for category, skills in self.COMMON_SKILLS.items():
            for skill in skills:
                skill_lower = skill.lower()
                synonyms[skill_lower].add(skill_lower)
                # Add common variations
                if ' ' in skill_lower:
                    synonyms[skill_lower.replace(' ', '')].add(skill_lower)
                    synonyms[skill_lower.replace(' ', '-')].add(skill_lower)
                # Add common abbreviations
                if skill_lower.startswith('microsoft'):
                    synonyms['ms'].add(skill_lower)
                if skill_lower.startswith('amazon'):
                    synonyms['aws'].add(skill_lower)
        return synonyms

    def _find_potential_matches(self, text: str, trie: Dict) -> Dict[str, List[int]]:
        """Find potential skill matches using trie with word boundary checks."""
        matches = defaultdict(list)
        text_lower = text.lower()
        
        # Split text into words while preserving positions
        words = []
        for match in re.finditer(r'\b\w+\b', text_lower):
            words.append((match.group(), match.start()))
        
        for word, start_pos in words:
            current = trie
            j = 0
            while j < len(word) and word[j] in current:
                current = current[word[j]]
                if '__end__' in current:
                    # Only add if it's a complete word match
                    if j == len(word) - 1:  # We've matched the entire word
                        matches[word].append(start_pos)
                j += 1
                
        return matches

    def _calculate_advanced_confidence(
        self,
        skill: str,
        section: str,
        context: str,
        positions: List[int],
        ngrams: Set[str],
        skill_ngrams: Dict[str, Set[str]],
        skill_synonyms: Dict[str, Set[str]]
    ) -> float:
        """Calculate advanced confidence score using multiple factors."""
        # Base confidence from section
        base_scores = {
            'skills_section': 0.9,
            'experience_section': 0.8,
            'education_section': 0.7,
            'certification_section': 0.85,
            'project_section': 0.75,
            'full_text': 0.6  # Lower base score for full text matches
        }
        confidence = base_scores.get(section, 0.5)

        # N-gram similarity score
        skill_grams = set()
        for n in range(2, 5):
            for i in range(len(skill) - n + 1):
                skill_grams.add(skill[i:i+n])

        ngram_similarity = len(skill_grams & ngrams) / len(skill_grams) if skill_grams else 0
        confidence += ngram_similarity * 0.2

        # Context analysis
        for pos in positions:
            # Check surrounding context
            start = max(0, pos - 50)
            end = min(len(context), pos + 50)
            local_context = context[start:end]

            # Check for proficiency indicators
            proficiency_terms = {
                'expert': 0.2,
                'proficient': 0.15,
                'skilled': 0.1,
                'experienced': 0.1,
                'advanced': 0.15,
                'strong': 0.1,
                'extensive': 0.1,
                'comprehensive': 0.1
            }

            for term, boost in proficiency_terms.items():
                if term in local_context.lower():
                    confidence += boost
                    break

        # Synonym matching
        if skill in skill_synonyms:
            for synonym in skill_synonyms[skill]:
                if synonym in context.lower():
                    confidence += 0.1
                    break

        # Position-based confidence
        if len(positions) > 1:
            confidence += min(0.1 * (len(positions) - 1), 0.3)

        # Boost confidence for exact matches in skills section
        if section == 'skills_section' and skill.lower() in context.lower():
            confidence += 0.2

        # Cap confidence between 0.1 and 1.0
        return max(0.1, min(1.0, confidence))

    def _is_duplicate_skill(self, new_skill: Dict, existing_skill: Dict) -> bool:
        """Check if a skill is a duplicate using advanced comparison."""
        # Check exact match
        if new_skill["name"] == existing_skill["name"]:
            return True
        
        # Check for similar names using Levenshtein distance
        if self._levenshtein_ratio(new_skill["name"], existing_skill["name"]) > 0.8:
            return True
        
        # Check for overlapping positions
        new_positions = set(new_skill["positions"])
        existing_positions = set(existing_skill["positions"])
        if new_positions & existing_positions:
            return True
        
        return False

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity ratio."""
        if not s1 or not s2:
            return 0.0
        
        # Convert to lowercase for better matching
        s1, s2 = s1.lower(), s2.lower()
        
        # Calculate Levenshtein distance
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if not s2:
            return 0.0
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        # Calculate similarity ratio
        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 0.0

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text

    def _generate_ngrams(self, text: str) -> Set[str]:
        """Generate n-grams from text."""
        ngrams = set()
        words = text.split()
        
        # Generate word n-grams
        for n in range(2, 4):
            for i in range(len(words) - n + 1):
                ngrams.add(' '.join(words[i:i+n]))
        
        # Generate character n-grams
        for n in range(2, 5):
            for i in range(len(text) - n + 1):
                ngrams.add(text[i:i+n])
        
        return ngrams

    def _extract_email(self, text: str) -> ExtractedValue:
        """Extract email address"""
        # Try regex pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        if match:
            return ExtractedValue(match.group(0), 0.9, "regex")
        return ExtractedValue("", 0.0, "none")

    def _extract_phone(self, text: str) -> ExtractedValue:
        """Extract phone number with improved pattern matching"""
        # Try regex patterns for different phone formats
        phone_patterns = [
            # Format: (XXX) XXX-XXXX
            r'(?:Phone|Tel|Mobile|Cell|Contact|Call)?[:\s]*\(?(\d{3})\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
            # Format: XXX-XXX-XXXX
            r'(?:Phone|Tel|Mobile|Cell|Contact|Call)?[:\s]*\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
            # Format: XXX.XXX.XXXX
            r'(?:Phone|Tel|Mobile|Cell|Contact|Call)?[:\s]*\d{3}\.\d{3}\.\d{4}',
            # Format: XXXXXXXXXX
            r'(?:Phone|Tel|Mobile|Cell|Contact|Call)?[:\s]*\b\d{10}\b',
            # Format: +1 XXX-XXX-XXXX
            r'(?:Phone|Tel|Mobile|Cell|Contact|Call)?[:\s]*\+1[\s\-\.]?\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
            # Format: 1-XXX-XXX-XXXX
            r'(?:Phone|Tel|Mobile|Cell|Contact|Call)?[:\s]*1[\s\-\.]?\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the full phone number including any formatting
                phone = match.group(0)
                # Clean up the phone number to just digits
                clean_phone = re.sub(r'[^\d]', '', phone)
                if len(clean_phone) == 10:  # Must be 10 digits
                    return ExtractedValue(clean_phone, 0.9, "regex")
                elif len(clean_phone) == 11 and clean_phone.startswith('1'):  # Handle country code
                    return ExtractedValue(clean_phone[1:], 0.9, "regex")
        
        return ExtractedValue("", 0.0, "none")

    def _is_contact_info(self, text: str) -> bool:
        """Helper to check if a string contains phone number or email patterns."""
        phone_email_pattern = re.compile(r'''(?:\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b)''', re.IGNORECASE)
        return bool(phone_email_pattern.search(text))

    def _extract_contact_info(self, text: str) -> Dict[str, ExtractedValue]:
        """Extract contact information including email and phone"""
        contact_info = {}
        
        # Extract primary email
        primary_email = self._extract_email(text)
        contact_info["primary_email"] = primary_email
        
        # Extract phone number
        phone = self._extract_phone(text)
        contact_info["phone"] = phone
        
        # Extract secondary email if present
        secondary_email_pattern = r'(?:Secondary|Alternate|Other)\s+Email[:\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
        secondary_match = re.search(secondary_email_pattern, text, re.IGNORECASE)
        if secondary_match:
            contact_info["secondary_email"] = ExtractedValue(secondary_match.group(1), 0.8, "regex")
        else:
            contact_info["secondary_email"] = ExtractedValue("", 0.0, "none")
        
        return contact_info

    def _extract_name_and_location(self, text: str) -> Dict[str, ExtractedValue]:
        """Extract name and location information"""
        # Extract name
        name = self._extract_name(text)
        
        # Extract location
        location = self._extract_location(text)
        
        # Split name into first and last name if it exists
        first_name = ""
        last_name = ""
        if name.value:
            name_parts = name.value.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:])
            else:
                # If only one name part, use it as first name
                first_name = name.value
        
        return {
            "first_name": ExtractedValue(first_name, name.confidence, name.method),
            "last_name": ExtractedValue(last_name, name.confidence, name.method),
            "city": location.get("city", ExtractedValue("", 0.0, "none")),
            "state": location.get("state", ExtractedValue("", 0.0, "none")),
            "zip": location.get("zip", ExtractedValue("", 0.0, "none"))
        }

    def _extract_work_authority(self, text: str) -> ExtractedValue:
        """Extract work authorization information"""
        try:
            # Common work authorization patterns
            patterns = [
                r'(?:Work Auth|Work Authorization|Authorization|Visa)[:\s]+([A-Za-z\s]+)',
                r'(?:Citizenship|Citizen)[:\s]+([A-Za-z\s]+)',
                r'(?:Visa Status|Status)[:\s]+([A-Za-z\s]+)',
                r'(?:Work Authorization|Authorization|Visa)[:\s]*is\s+([A-Za-z\s]+)',
                r'(?:Citizenship|Citizen)[:\s]*is\s+([A-Za-z\s]+)',
                r'(?:Visa Status|Status)[:\s]*is\s+([A-Za-z\s]+)',
                r'(?:Work Authorization|Authorization|Visa)[:\s]*-?\s*([A-Za-z\s]+)',
                r'(?:Citizenship|Citizen)[:\s]*-?\s*([A-Za-z\s]+)',
                r'(?:Visa Status|Status)[:\s]*-?\s*([A-Za-z\s]+)'
            ]
            
            # Look in the first 2000 characters (summary and header sections)
            summary_text = text[:2000]
            for pattern in patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    auth = match.group(1).strip()
                    # Normalize common variations
                    auth = auth.lower()
                    if 'green card' in auth or 'gc' in auth:
                        return ExtractedValue("Green Card", 0.9, "regex")
                    elif 'citizen' in auth:
                        return ExtractedValue("US Citizen", 0.9, "regex")
                    elif 'h1' in auth or 'h-1' in auth:
                        return ExtractedValue("H1B", 0.9, "regex")
                    elif 'h4' in auth or 'h-4' in auth:
                        return ExtractedValue("H4", 0.9, "regex")
                    elif 'l1' in auth or 'l-1' in auth:
                        return ExtractedValue("L1", 0.9, "regex")
                    elif 'l2' in auth or 'l-2' in auth:
                        return ExtractedValue("L2", 0.9, "regex")
                    elif 'ead' in auth:
                        return ExtractedValue("EAD", 0.9, "regex")
                    elif 'opt' in auth:
                        return ExtractedValue("OPT", 0.9, "regex")
                    elif 'cpt' in auth:
                        return ExtractedValue("CPT", 0.9, "regex")
                    else:
                        return ExtractedValue(auth.title(), 0.8, "regex")
            
            return ExtractedValue("", 0.0, "none")
            
        except Exception as e:
            logger.error(f"Error extracting work authorization: {e}")
            return ExtractedValue("", 0.0, "none")

    def _extract_tax_term(self, text: str) -> ExtractedValue:
        """Extract tax term (W2, C2C, 1099, contract, full time, etc.) from resume text"""
        try:
            # Search for tax terms in the first 2000 characters
            search_text = text[:2000].lower()
            for term in US_TAX_TERMS:
                # Use word boundaries for short terms
                if len(term) <= 4:
                    pattern = rf'\b{re.escape(term)}\b'
                else:
                    pattern = re.escape(term)
                match = re.search(pattern, search_text)
                if match:
                    return ExtractedValue(term.upper(), 0.9, "regex")
            return ExtractedValue("", 0.0, "none")
        except Exception as e:
            logger.error(f"Error extracting tax term: {e}")
            return ExtractedValue("", 0.0, "none")

    def _normalize_skill(self, skill: str) -> str:
        """Normalize individual skill names for better matching."""
        if not skill:
            return ""

        # Convert to lowercase
        skill = skill.lower()

        # Handle common skill prefixes and suffixes
        prefixes = {
            r'^expert\s+in\s+': '',
            r'^proficient\s+in\s+': '',
            r'^skilled\s+in\s+': '',
            r'^experienced\s+in\s+': '',
            r'^advanced\s+': '',
            r'^basic\s+': '',
            r'^intermediate\s+': '',
            r'^beginner\s+': '',
            r'^novice\s+': '',
            r'^expert\s+': '',
            r'^professional\s+': '',
            r'^senior\s+': '',
            r'^junior\s+': '',
            r'^lead\s+': '',
            r'^principal\s+': '',
            r'^chief\s+': '',
            r'^head\s+of\s+': '',
            r'^director\s+of\s+': '',
            r'^manager\s+of\s+': '',
            r'^specialist\s+in\s+': ''
        }

        for pattern, replacement in prefixes.items():
            skill = re.sub(pattern, replacement, skill)

        # Handle common skill suffixes
        suffixes = {
            r'\s+expert$': '',
            r'\s+professional$': '',
            r'\s+specialist$': '',
            r'\s+engineer$': '',
            r'\s+developer$': '',
            r'\s+administrator$': '',
            r'\s+analyst$': '',
            r'\s+consultant$': '',
            r'\s+architect$': '',
            r'\s+manager$': '',
            r'\s+lead$': '',
            r'\s+senior$': '',
            r'\s+junior$': '',
            r'\s+associate$': '',
            r'\s+principal$': '',
            r'\s+chief$': '',
            r'\s+head$': '',
            r'\s+director$': '',
            r'\s+specialist$': '',
            r'\s+expert$': ''
        }

        for pattern, replacement in suffixes.items():
            skill = re.sub(pattern, replacement, skill)

        # Handle common skill variations
        variations = {
            r'\bprogramming\b': ['\bcoding\b', '\bdevelopment\b', '\bsoftware development\b'],
            r'\bframework\b': ['\blibrary\b', '\bplatform\b', '\btoolkit\b'],
            r'\bdatabase\b': ['\bdb\b', '\bdatastore\b', '\bdata store\b'],
            r'\bcloud\b': ['\bcloud computing\b', '\bcloud platform\b'],
            r'\bdevops\b': ['\bdevelopment operations\b', '\bdev ops\b'],
            r'\bmethodology\b': ['\bmethod\b', '\bapproach\b', '\bprocess\b'],
            r'\banalysis\b': ['\banalytics\b', '\banalyzing\b', '\banalyze\b'],
            r'\bmanagement\b': ['\bmanaging\b', '\bmanage\b', '\badminister\b'],
            r'\bdevelopment\b': ['\bdeveloping\b', '\bdevelop\b', '\bdev\b'],
            r'\bdesign\b': ['\bdesigning\b', '\barchitect\b', '\barchitecture\b'],
            r'\bimplementation\b': ['\bimplementing\b', '\bimplement\b', '\bdeploy\b'],
            r'\btesting\b': ['\btest\b', '\bqa\b', '\bquality assurance\b'],
            r'\bsecurity\b': ['\bsec\b', '\bcybersecurity\b', '\bcyber security\b'],
            r'\bnetworking\b': ['\bnetwork\b', '\bnetworks\b', '\bnetwork engineering\b'],
            r'\badministration\b': ['\badmin\b', '\bsystem administration\b', '\bsysadmin\b'],
            r'\bengineering\b': ['\bengineer\b', '\beng\b', '\btechnical\b'],
            r'\bconsulting\b': ['\bconsultant\b', '\bconsult\b', '\badvisory\b'],
            r'\barchitecture\b': ['\barchitect\b', '\barch\b', '\bsystem design\b'],
            r'\boperations\b': ['\bops\b', '\boperational\b', '\bopex\b'],
            r'\bstrategy\b': ['\bstrategic\b', '\bstrategic planning\b', '\bplanning\b']
        }

        for base, vars in variations.items():
            pattern = r'\b(' + '|'.join(vars) + r')\b'
            skill = re.sub(pattern, base, skill)

        # Handle common abbreviations
        abbreviations = {
            r'\bms\b': 'microsoft',
            r'\baws\b': 'amazon web services',
            r'\bazure\b': 'microsoft azure',
            r'\bgcp\b': 'google cloud platform',
            r'\bdevops\b': 'devops',
            r'\bci/cd\b': 'continuous integration continuous deployment',
            r'\bui/ux\b': 'user interface user experience',
            r'\bapi\b': 'application programming interface',
            r'\bui\b': 'user interface',
            r'\bux\b': 'user experience',
            r'\bqa\b': 'quality assurance',
            r'\bpm\b': 'project management',
            r'\bhr\b': 'human resources',
            r'\bit\b': 'information technology',
            r'\bml\b': 'machine learning',
            r'\bai\b': 'artificial intelligence',
            r'\bdb\b': 'database',
            r'\bsql\b': 'structured query language',
            r'\bnosql\b': 'not only sql'
        }

        for pattern, replacement in abbreviations.items():
            skill = re.sub(pattern, replacement, skill)

        # Final cleanup
        skill = re.sub(r'\s+', ' ', skill)  # Normalize whitespace
        skill = skill.strip()

        return skill

    def _extract_education(self, text: str) -> ExtractedValue:
        """Extract education information from resume text."""
        if not text:
            return ExtractedValue([], 0.0, "none")

        education_list = []
        confidence = 0.0

        # Common education section headers
        education_headers = [
            r"(?i)education[:|\n]",
            r"(?i)academic\s+background[:|\n]",
            r"(?i)educational\s+qualifications[:|\n]",
            r"(?i)academic\s+qualifications[:|\n]"
        ]

        # Find education section
        education_text = ""
        for header in education_headers:
            match = re.search(header + r"(.*?)(?:\n\n|\Z)", text, re.DOTALL)
            if match:
                education_text = match.group(1)
                break

        if education_text:
            # Split into individual education entries
            entries = re.split(r'\n(?=[A-Z][a-z]|\d{4})', education_text)
            
            for entry in entries:
                if not entry.strip():
                    continue

                education_info = {}
                
                # Extract degree
                degree_patterns = [
                    r"(?i)(bachelor|master|phd|doctorate|associate|diploma|certificate|b\.s\.|m\.s\.|b\.a\.|m\.a\.|b\.e\.|m\.e\.|b\.tech|m\.tech|b\.sc\.|m\.sc\.)[^,\n]*",
                    r"(?i)(bs|ms|ba|ma|be|me|btech|mtech|bsc|msc)[^,\n]*"
                ]
                
                for pattern in degree_patterns:
                    degree_match = re.search(pattern, entry)
                    if degree_match:
                        education_info['degree'] = degree_match.group(0).strip()
                        break

                # Extract major/specialization
                major_patterns = [
                    r"(?:in|of|majoring in|specializing in)\s+([^,\n]+)",
                    r"(?:with|with a focus on|with specialization in)\s+([^,\n]+)"
                ]
                
                for pattern in major_patterns:
                    major_match = re.search(pattern, entry, re.IGNORECASE)
                    if major_match:
                        education_info['major'] = major_match.group(1).strip()
                        break

                # Extract institution
                institution_patterns = [
                    r"(?:from|at)\s+([^,\n]+(?:university|college|institute|school))",
                    r"([^,\n]+(?:university|college|institute|school))"
                ]
                
                for pattern in institution_patterns:
                    inst_match = re.search(pattern, entry, re.IGNORECASE)
                    if inst_match:
                        education_info['institution'] = inst_match.group(1).strip()
                        break

                # Extract year/date
                year_patterns = [
                    r"(?:graduated|completed|earned|obtained|received)\s+(?:in\s+)?(\d{4})",
                    r"(?:class of|batch of)\s+(\d{4})",
                    r"(?:from|during)\s+(\d{4})\s*(?:to|-)?\s*(\d{4})?",
                    r"(\d{4})\s*(?:to|-)?\s*(\d{4})?"
                ]
                
                for pattern in year_patterns:
                    year_match = re.search(pattern, entry)
                    if year_match:
                        if year_match.group(2):  # If there's a range
                            education_info['start_year'] = year_match.group(1)
                            education_info['end_year'] = year_match.group(2)
                        else:
                            education_info['year'] = year_match.group(1)
                        break

                # Extract GPA if present
                gpa_match = re.search(r"(?:gpa|grade point average)[:\s]+(\d\.\d{1,2})", entry, re.IGNORECASE)
                if gpa_match:
                    education_info['gpa'] = gpa_match.group(1)

                if education_info:
                    education_list.append(education_info)

        # Calculate confidence based on number of fields extracted
        if education_list:
            total_fields = sum(len(edu) for edu in education_list)
            confidence = min(0.9, 0.3 + (total_fields * 0.1))

        return ExtractedValue(education_list, confidence, "multi_method")

    def _extract_certifications(self, text: str) -> ExtractedValue:
        """Extract certifications from resume text."""
        if not text:
            return ExtractedValue([], 0.0, "none")

        certifications = []
        confidence = 0.0

        # Common certification section headers
        cert_headers = [
            r"(?i)certifications?[:|\n]",
            r"(?i)certificates?[:|\n]",
            r"(?i)professional\s+certifications?[:|\n]",
            r"(?i)technical\s+certifications?[:|\n]",
            r"(?i)industry\s+certifications?[:|\n]"
        ]

        # Find certification section
        cert_text = ""
        for header in cert_headers:
            match = re.search(header + r"(.*?)(?:\n\n|\Z)", text, re.DOTALL)
            if match:
                cert_text = match.group(1)
                break

        if cert_text:
            # Split into individual certification entries
            entries = re.split(r'\n(?=[A-Z]|\d{4})', cert_text)
            
            for entry in entries:
                if not entry.strip():
                    continue

                cert_info = {}
                
                # Extract certification name
                cert_patterns = [
                    r"([A-Z][A-Za-z\s]+(?:certified|professional|associate|specialist|expert|master|foundation|practitioner|architect|administrator|engineer|developer|analyst|consultant))",
                    r"([A-Z][A-Za-z\s]+(?:certification|certificate|certified))"
                ]
                
                for pattern in cert_patterns:
                    cert_match = re.search(pattern, entry)
                    if cert_match:
                        cert_info['name'] = cert_match.group(1).strip()
                        break

                # Extract issuing organization
                org_patterns = [
                    r"(?:from|by|issued by)\s+([^,\n]+)",
                    r"([^,\n]+(?:microsoft|aws|google|oracle|cisco|comptia|pmi|itil|red\s+hat|vmware|salesforce|ibm|adobe|apple|linux|amazon))"
                ]
                
                for pattern in org_patterns:
                    org_match = re.search(pattern, entry, re.IGNORECASE)
                    if org_match:
                        cert_info['issuer'] = org_match.group(1).strip()
                        break

                # Extract date
                date_patterns = [
                    r"(?:obtained|earned|received|completed|issued)\s+(?:in\s+)?(\d{4})",
                    r"(?:valid|expires?)\s+(?:until|till|through|to)?\s+(\d{4})",
                    r"(\d{4})"
                ]
                
                for pattern in date_patterns:
                    date_match = re.search(pattern, entry)
                    if date_match:
                        cert_info['year'] = date_match.group(1)
                        break

                # Extract ID if present
                id_match = re.search(r"(?:id|number|#)[:\s]+([A-Z0-9-]+)", entry, re.IGNORECASE)
                if id_match:
                    cert_info['id'] = id_match.group(1)

                if cert_info:
                    certifications.append(cert_info)

        # Calculate confidence based on number of fields extracted
        if certifications:
            total_fields = sum(len(cert) for cert in certifications)
            confidence = min(0.9, 0.3 + (total_fields * 0.1))

        return ExtractedValue(certifications, confidence, "multi_method")